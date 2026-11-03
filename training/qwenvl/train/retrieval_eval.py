"""Retrieval validation during training (OMNI_HTSTEP_LORA.md 4.4).

Every item of a manifest is encoded at batch 1 (padding changes the embedding, GUIDE_TRAIN_EVAL_RETRIEVAL.md) through
the training loader and the model's ``pred_embeds`` path -- what ``scripts/eval_youcookii.py`` does -- and scored
N -> N on the diagonal, text -> AV (t2v) and AV -> text (v2t): R@1/5/10, NDCG@10, MRR
(``omniretriever.evaluation.metrics``), plus R@1 multi-positive (any clip / text with the same step text counts).
The checkpoint metric is t2v R@1 of the paper protocol (diagonal).

The callback evaluates the untrained model first (the pt baseline, embeddings kept), then every ``every`` optimiser
steps; keeps the LoRA tensors of the best evaluation in ``<output_dir>/best`` (the heads stay the pt ones: they are
frozen with LORA_ONLY); stops after ``patience`` evaluations without a better R@1; writes ``metrics.json`` and uploads
both to ``hf_run_repo``. Its state lives in ``eval_state.json`` so a resumed run keeps counting.
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from transformers import TrainerCallback

from omniretriever.evaluation.metrics import all_metrics

LORA_FILE = "adapter_model.safetensors"


def eval_dataset(manifest: str, data_args, tokenizer):
    """A LazySupervisedDataset of ``manifest`` that returns None instead of a random replacement on error."""
    from qwenvl.data.data_qwen import LazySupervisedDataset

    args = copy.copy(data_args)
    args.dataset_use = manifest
    args.run_test = True
    return LazySupervisedDataset(tokenizer=tokenizer, data_args=args)


def _to_device(batch: dict, device, dtype) -> dict:
    out = {}
    for k, v in batch.items():
        if v is None or k in ("texts", "ids"):
            continue
        if isinstance(v, torch.Tensor):
            if k in ("pixel_values", "pixel_values_videos", "input_features") and v.dtype == torch.float32:
                out[k] = v.to(device, dtype=dtype)
            else:
                out[k] = v.to(device)
        elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
            out[k] = [t.to(device) for t in v]
        else:
            out[k] = v
    return out


@torch.no_grad()
def encode(model, dataset, max_items: int = 0) -> dict:
    """{'text', 'av': (N, D) float32 L2-normalised, 'ids', 'texts', 'failed': [ids]} at batch 1."""
    from qwenvl.data.data_qwen import DataCollatorForOmniDataset

    collate = DataCollatorForOmniDataset()
    was_training = model.training
    model.eval()
    p = next(model.parameters())
    txt, av, ids, texts, failed = [], [], [], [], []
    n = len(dataset) if not max_items else min(max_items, len(dataset))
    try:
        for i in range(n):
            rec = dataset.list_data_dict[i]
            item = dataset._get_item(i)
            if item is None:
                failed.append(rec.get("id"))
                continue
            batch = _to_device(collate([item]), p.device, p.dtype)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=p.device.type == "cuda"):
                out = model(**batch, output_hidden_states=True, return_dict=True, pred_embeds=True)
            av.append(torch.nn.functional.normalize(out.mllm_embeds.float(), dim=-1).cpu())
            txt.append(torch.nn.functional.normalize(out.text_embeds.float(), dim=-1).cpu())
            ids.append(rec.get("id"))
            texts.append(rec.get("text"))
    finally:
        model.train(was_training)
    return {"text": torch.cat(txt).numpy(), "av": torch.cat(av).numpy(), "ids": ids, "texts": texts,
            "failed": failed}


def multi_positive_r1(sim: np.ndarray, texts) -> float:
    """Share of rows whose top-1 column has the same text as the row (diagonal included)."""
    t = np.asarray(texts, dtype=object)
    return float(np.mean(t[sim.argmax(1)] == t))


def score(emb: dict) -> dict:
    sim = emb["text"] @ emb["av"].T                      # rows: text queries, columns: AV clips
    out = {}
    for name, s in (("t2v", sim), ("v2t", sim.T)):
        m = all_metrics(s)
        out[name] = {k: round(v * 100, 3) if k != "median_rank" else v for k, v in m.items()}
        out[name]["R@1_multi"] = round(100 * multi_positive_r1(s, emb["texts"]), 3)
    out["n"] = int(len(emb["ids"]))
    out["failed"] = len(emb["failed"])
    return out


def save_embeddings(emb: dict, path: str) -> None:
    np.savez(path, text=emb["text"].astype(np.float16), av=emb["av"].astype(np.float16),
             ids=np.array(emb["ids"]), texts=np.array(emb["texts"]))


def save_lora(model, out_dir: str) -> None:
    """Only the lora_A / lora_B tensors (+ adapter_config.json); the heads are the pt adapter's."""
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file

    os.makedirs(out_dir, exist_ok=True)
    sd = {k: v.detach().cpu().contiguous() for k, v in get_peft_model_state_dict(model).items() if "lora_" in k}
    save_file(sd, os.path.join(out_dir, "lora_only.safetensors"))
    model.peft_config["default"].save_pretrained(out_dir)
    Path(out_dir, "README.md").write_text(
        "LoRA tensors only (lora_only.safetensors). classify_linear / beats_ln / beats_proj were frozen "
        "(LORA_ONLY=True) and are those of the pt adapter: build a full adapter with "
        "`python scripts/materialize_adapter.py <pt adapter> <this dir> <out>`.\n", encoding="utf-8")


def materialize_adapter(pt_dir: str, lora_dir: str, out_dir: str) -> str:
    """pt adapter with its LoRA tensors replaced by those of ``lora_dir`` -> a loadable adapter in ``out_dir``."""
    from safetensors.torch import load_file, save_file

    pt = load_file(os.path.join(pt_dir, LORA_FILE))
    lora = load_file(os.path.join(lora_dir, "lora_only.safetensors"))
    missing = sorted(set(lora) - set(pt))
    if missing:
        raise KeyError(f"{len(missing)} LoRA keys not in the pt adapter, e.g. {missing[:3]}")
    pt.update(lora)
    os.makedirs(out_dir, exist_ok=True)
    save_file(pt, os.path.join(out_dir, LORA_FILE))
    shutil.copy(os.path.join(pt_dir, "adapter_config.json"), os.path.join(out_dir, "adapter_config.json"))
    return out_dir


class RetrievalEvalCallback(TrainerCallback):
    def __init__(self, model, val_dataset, output_dir: str, every: int, patience: int = 3,
                 track_dataset=None, hf_run_repo: str = "", run_name: str = ""):
        self.model, self.val, self.track = model, val_dataset, track_dataset
        self.dir, self.every, self.patience = output_dir, max(1, every), patience
        self.hf_run_repo, self.run_name = hf_run_repo, run_name or os.path.basename(output_dir.rstrip("/"))
        self.state_path = os.path.join(output_dir, "eval_state.json")
        self.st = {"best": None, "best_step": None, "bad_evals": 0, "history": [], "track": []}
        if os.path.isfile(self.state_path):
            self.st = json.load(open(self.state_path, encoding="utf-8"))

    def _save_state(self):
        os.makedirs(self.dir, exist_ok=True)
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(self.st, f, indent=1)
        with open(os.path.join(self.dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump({"run": self.run_name, "metric": "t2v R@1 (paper protocol, diagonal)", **self.st}, f, indent=1)

    def _upload(self):
        if not self.hf_run_repo:
            return
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            api.create_repo(self.hf_run_repo, repo_type="model", private=True, exist_ok=True)
            api.upload_file(path_or_fileobj=os.path.join(self.dir, "metrics.json"), repo_id=self.hf_run_repo,
                            path_in_repo=f"runs/{self.run_name}/metrics.json")
            best = os.path.join(self.dir, "best")
            if os.path.isdir(best):
                api.upload_folder(folder_path=best, repo_id=self.hf_run_repo, path_in_repo=f"runs/{self.run_name}/best")
        except Exception as e:  # the run must survive a network error
            print(f"[eval] upload to {self.hf_run_repo} failed: {e}", flush=True)

    def _evaluate(self, step: int, epoch: float):
        t0 = time.time()
        emb = encode(self.model, self.val)
        m = {"step": step, "epoch": round(float(epoch or 0), 4), **score(emb), "seconds": round(time.time() - t0, 1)}
        if emb["failed"]:
            print(f"[eval] FAIL: {len(emb['failed'])} val items could not be loaded: {emb['failed'][:5]}", flush=True)
        r1 = m["t2v"]["R@1"]
        self.st["history"].append(m)
        if step == 0:
            save_embeddings(emb, os.path.join(self.dir, "val_embeds_pt.npz"))
        if self.st["best"] is None or r1 > self.st["best"]:
            self.st.update(best=r1, best_step=step, bad_evals=0)
            if step > 0:
                save_lora(self.model, os.path.join(self.dir, "best"))
                save_embeddings(emb, os.path.join(self.dir, "val_embeds_best.npz"))
        elif step > 0:
            self.st["bad_evals"] += 1
        print(f"[eval] step {step} t2v R@1 {r1:.2f} (multi {m['t2v']['R@1_multi']:.2f}) v2t R@1 {m['v2t']['R@1']:.2f}"
              f" | best {self.st['best']:.2f} @ {self.st['best_step']} | {m['seconds']:.0f}s", flush=True)
        self._save_state()
        self._upload()

    def on_train_begin(self, args, state, control, **kw):
        if not self.st["history"]:          # not on a resumed run: the pt baseline
            self._evaluate(0, 0.0)
            self._track(0, 0.0)

    def on_step_end(self, args, state, control, **kw):
        last = state.global_step >= state.max_steps
        if state.global_step % self.every == 0 or last:
            if any(h["step"] == state.global_step for h in self.st["history"]):
                return
            self._evaluate(state.global_step, state.epoch)
            control.should_save = True               # a checkpoint after every eval (Colab sessions get cut)
            if self.st["bad_evals"] >= self.patience:
                print(f"[eval] early stop: {self.patience} evals without a better t2v R@1", flush=True)
                control.should_training_stop = True

    def _track(self, step: int, epoch: float):
        """Forgetting track (benchmark subset): logged, never used to pick anything."""
        if self.track is None:
            return
        t0 = time.time()
        m = score(encode(self.model, self.track))
        self.st["track"].append({"step": step, "epoch": round(float(epoch or 0), 4), "t2av": m["t2v"],
                                 "av2t": m["v2t"], "n": m["n"], "seconds": round(time.time() - t0, 1)})
        print(f"[track] step {step} t2av R@1 {m['t2v']['R@1']:.2f} av2t R@1 {m['v2t']['R@1']:.2f}", flush=True)
        self._save_state()
        self._upload()

    def on_epoch_end(self, args, state, control, **kw):
        self._track(state.global_step, state.epoch)
