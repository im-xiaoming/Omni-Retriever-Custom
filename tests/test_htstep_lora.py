"""Unit tests of the HT-Step LoRA pipeline (plan OMNI_HTSTEP_LORA.md 4.8). CPU only.

    python -m pytest tests/test_htstep_lora.py -q
"""
import csv
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (os.path.join(ROOT, "scripts"), os.path.join(ROOT, "training"), os.path.join(ROOT, "src")):
    sys.path.insert(0, p)

import convert_htstep as ch  # noqa: E402
from qwenvl.train import grad_cache as gc  # noqa: E402

HTSTEP = "D:/Học/KL/Data/Htstep/htstep/data"
DATA = os.path.join(ROOT, "data", "htstep")


# ------------------------------------------------------------------------------------------------- selection rules
def toy():
    tax = {i: {"global_step_index": str(i), "activity": f"Act{i % 3}", "headline": f" Step {i}. ", "variation": "A"}
           for i in range(60)}
    seg = lambda s, e, i, p="Partial Match": {"segment": [s, e], "id": i, "partial": p}  # noqa: E731
    ann = {
        "v_long": {"subset": "train", "duration": 999.0,
                   "annotations": [seg(5, 5, 0)] + [seg(i, i + 1, i % 60, "Full Match") for i in range(50)]},
        "v_val": {"subset": "val_seen", "duration": 50.0, "annotations": [seg(3, 9, 7), seg(1, 2, 8)]},
        "v_yc2": {"subset": "train", "duration": 50.0, "annotations": [seg(1, 2, 1)]},
        "v_nomp4": {"subset": "train", "duration": 50.0, "annotations": [seg(1, 2, 1)]},
        "v_test": {"subset": "test_seen", "duration": 50.0, "annotations": [seg(1, 2, 1)]},
    }
    src = {v: [{"repo": "r", "tar": "videos/htstep_videos_1000.tar", "member": f"{v}.mp4"}]
           for v in ("v_long", "v_val", "v_yc2")}
    src["v_val"].insert(0, {"repo": ch.VAL_VIDEOS_REPO, "file": "part1/v_val.mp4"})
    return ann, tax, src, {"v_yc2"}


def test_selection_rules():
    ann, tax, src, yc2 = toy()
    videos, items, excluded, counts = ch.select(ann, tax, src, yc2)
    assert set(videos["train"]) == {"v_long"} and set(videos["val"]) == {"v_val"}
    assert excluded == {"v_yc2": "yc2_overlap", "v_nomp4": "no_mp4"}          # test_seen ignored
    long = [it for it in items["train"] if it["video_id"] == "v_long"]
    assert len(long) == 48                                                     # end <= start dropped, then 48
    assert [it["step_idx"] for it in long] == list(range(48))
    assert long[0]["segment"] == [0.0, 1.0] and long[-1]["segment"] == [47.0, 48.0]   # json order
    val = items["val"]
    assert [it["segment"] for it in val] == [[3.0, 9.0], [1.0, 2.0]]           # not re-sorted
    assert {it["partial"] for it in long} == {"Full Match"} and val[0]["partial"] == "Partial Match"
    assert videos["val"]["v_val"]["source"]["repo"] == ch.VAL_VIDEOS_REPO     # the val repo copy first
    assert counts["train"]["steps_after_step_filter"] == 48 + 1 + 1


def test_text_and_key():
    ann, tax, src, yc2 = toy()
    _, items, _, _ = ch.select(ann, tax, src, yc2)
    it = items["val"][0]
    assert it["text"] == "Act1: Step 7."                                       # activity of the taxonomy row, strip
    assert it["id"] == "v_val#0" and it["global_step_index"] == 7
    rec = ch.build_record(it, [3.0, 8.5], 5.5)
    assert rec["conversations"][1]["value"] == rec["text"] == "Act1: Step 7."
    assert rec["timestamps"] == rec["window"] == [3.0, 8.5] and rec["segment"] == [3.0, 9.0]
    assert rec["clip"] == "clips/v_val/0.npz" and rec["video"] == "v_val.mp4"


def test_split_reproducible():
    ann, tax, src, yc2 = toy()
    assert json.dumps(ch.select(ann, tax, src, yc2), sort_keys=True) == json.dumps(
        ch.select(ann, tax, src, yc2), sort_keys=True)
    assert ch.same_text_pairs(["a"] * 40 + ["b"] * 40, 32, 50) == ch.same_text_pairs(["a"] * 40 + ["b"] * 40, 32, 50)


def test_transcode_check():
    from omniretriever.data import htstep_io as io
    assert io.same_video({"frames": 10, "fps": "30/1", "duration": 0.333}, {"frames": 10, "fps": "30/1", "duration": 0.35})
    assert not io.same_video({"frames": 10, "fps": "30/1", "duration": 0.3}, {"frames": 9, "fps": "30/1", "duration": 0.3})
    assert not io.same_video({"frames": 10, "fps": "30/1", "duration": 0.3}, {"frames": 10, "fps": "25/1", "duration": 0.3})


# ------------------------------------------------------------------------------------------------- mask + loss
def test_mask_and_loss_by_hand():
    texts = ["x", "y", "x"]
    m = gc.same_text_mask(texts)
    assert m.tolist() == [[False, False, True], [False, False, False], [True, False, False]]
    torch.manual_seed(0)
    av, tx = torch.randn(3, 4), torch.randn(3, 4)
    loss, v2t, t2v = gc.pair_loss(av, tx, 0.5, m)
    a = (av / av.norm(dim=1, keepdim=True)).numpy()
    t = (tx / tx.norm(dim=1, keepdim=True)).numpy()

    def ce(s, mask):
        s = np.where(mask, -np.inf, s)
        return float(np.mean([-s[i, i] + np.log(np.exp(s[i]).sum()) for i in range(len(s))]))
    mk = m.numpy()
    assert abs(v2t.item() - ce(a @ t.T / 0.5, mk)) < 1e-5
    assert abs(t2v.item() - ce(t @ a.T / 0.5, mk.T)) < 1e-5
    assert abs(loss.item() - 0.5 * (v2t.item() + t2v.item())) < 1e-6
    _, v2t0, _ = gc.pair_loss(av, tx, 0.5, None)
    assert v2t0.item() != v2t.item()


# ------------------------------------------------------------------------------------------------- GradCache
class Tiny(torch.nn.Module):
    """embed() interface of the model: forward(**batch, ...) -> .mllm_embeds / .text_embeds; dropout on."""

    def __init__(self):
        super().__init__()
        self.f, self.g, self.drop = torch.nn.Linear(6, 5), torch.nn.Linear(4, 5), torch.nn.Dropout(0.3)

    def forward(self, av, txt, **kw):
        return SimpleNamespace(mllm_embeds=self.f(self.drop(av)), text_embeds=self.g(self.drop(txt)))


def test_grad_cache_matches_full_batch():
    torch.manual_seed(1)
    items = [{"av": torch.randn(6), "txt": torch.randn(4), "text": t} for t in "abcadbca"]
    collate = lambda xs: {"av": torch.stack([x["av"] for x in xs]), "txt": torch.stack([x["txt"] for x in xs])}  # noqa
    ident = lambda b: b  # noqa: E731
    model = Tiny().train()

    def reference(chunk):
        """The exact gradient of the same function: chunked forwards with graph, one backward."""
        model.zero_grad()
        avs, txs = [], []
        for i in range(0, len(items), chunk):
            a, t = gc.embed(model, collate(items[i:i + chunk]))
            avs.append(a)
            txs.append(t)
        loss, _, _ = gc.pair_loss(torch.cat(avs), torch.cat(txs), 0.05, gc.same_text_mask([x["text"] for x in items]))
        loss.backward()
        return loss.item(), [p.grad.clone() for p in model.parameters()]

    for chunk in (1, 3, 8):
        torch.manual_seed(7)
        ref_loss, ref = reference(chunk)
        model.zero_grad()
        torch.manual_seed(7)
        out = gc.grad_cache_step(model, items, collate, ident, chunk, 0.05, True)
        assert abs(out["loss"].item() - ref_loss) < 1e-6
        assert out["masked_pairs"] == 2 * (3 + 1 + 1)      # a x3 -> 6 ordered pairs, b x2 -> 2, c x2 -> 2
        for p, r in zip(model.parameters(), ref):
            assert float((p.grad - r).norm() / r.norm()) < 1e-5


# ------------------------------------------------------------------------------------------------- clip loader
def make_mp4(path, seconds=3.0, fps=10, w=96, h=64, sr=16000):
    av = pytest.importorskip("av")
    with av.open(path, "w") as c:
        vs = c.add_stream("mpeg4", rate=fps)
        vs.width, vs.height, vs.pix_fmt = w, h, "yuv420p"
        as_ = c.add_stream("aac", rate=sr)
        rng = np.random.default_rng(0)
        for i in range(int(seconds * fps)):
            img = (rng.integers(0, 255, (h, w, 3)) // 2 + i * 3).astype(np.uint8)
            for pkt in vs.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(pkt)
        t = np.arange(int(seconds * sr)) / sr
        wav = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        for k in range(0, len(wav), 1024):
            fr = av.AudioFrame.from_ndarray(wav[None, k:k + 1024], format="fltp", layout="mono")
            fr.sample_rate = sr
            for pkt in as_.encode(fr):
                c.mux(pkt)
        for s in (vs, as_):
            for pkt in s.encode(None):
                c.mux(pkt)


@pytest.fixture(scope="module")
def mp4(tmp_path_factory):
    pytest.importorskip("decord")
    p = str(tmp_path_factory.mktemp("v") / "toy.mp4")
    make_mp4(p)
    return p


@pytest.fixture(scope="module")
def image_processor():
    mod = pytest.importorskip("transformers.models.qwen2_vl.image_processing_qwen2_vl")
    ip = mod.Qwen2VLImageProcessor()
    ip.min_pixels = ip.max_pixels = 50176                      # what data_qwen sets
    ip.size = {"shortest_edge": 50176, "longest_edge": 50176}
    return ip


def test_clip_matches_mp4_path(mp4, image_processor, tmp_path):
    from decord import VideoReader
    from omniretriever.data.clip import cut_clip, frame_indices, load_clip
    from omniretriever.data.media import load_audio_segment

    c = cut_clip(mp4, 0.6, 2.4, image_processor)
    path = str(tmp_path / "c.npz")
    np.savez_compressed(path, **c)
    clip = load_clip(path)
    vr = VideoReader(mp4, num_threads=1)                       # data_qwen.video_decord, timestamps = window
    idx, video_length = frame_indices(len(vr), vr.get_avg_fps(), 0.6, 2.4, 8)
    raw = torch.from_numpy(vr.get_batch(idx).asnumpy())
    ref = image_processor(images=None, videos=raw, return_tensors="pt")
    got = image_processor(images=None, videos=torch.from_numpy(clip["frames"]), do_resize=False, return_tensors="pt")
    assert torch.equal(ref["video_grid_thw"], got["video_grid_thw"])
    assert torch.allclose(ref["pixel_values_videos"], got["pixel_values_videos"], atol=1e-6)
    assert clip["video_length"] == pytest.approx(video_length) and clip["n_frames"] == 8
    wav = load_audio_segment(mp4, 0.6, 2.4)
    assert clip["wav"].shape == wav.shape and np.abs(clip["wav"] - wav).max() < 1e-4


def test_clip_window_rules(mp4, image_processor):
    from omniretriever.data.clip import PastEnd, cut_clip

    c = cut_clip(mp4, 2.0, 99.0, image_processor)              # end past the file: clamped to its length
    assert c["window"][1] == pytest.approx(c["total_frames"] / c["src_fps"])
    with pytest.raises(PastEnd):
        cut_clip(mp4, 50.0, 60.0, image_processor)             # start >= length


def test_process_clip_in_data_qwen(mp4, image_processor, tmp_path):
    dq = pytest.importorskip("qwenvl.data.data_qwen")
    from omniretriever.data.clip import cut_clip

    path = str(tmp_path / "c.npz")
    np.savez_compressed(path, **cut_clip(mp4, 0.6, 2.4, image_processor))
    fake = SimpleNamespace(data_args=SimpleNamespace(image_processor=image_processor))
    pix, grid, spg, wav = dq.LazySupervisedDataset.process_clip(fake, path)
    assert grid.shape == (1, 3) and len(spg) == 1 and wav is not None


# ------------------------------------------------------------------------------------------------- precut
def toy_items(tmp_path):
    ann, tax, src, yc2 = toy()
    _, items, _, _ = ch.select(ann, tax, src, yc2)
    paths = []
    for s in ("train", "val"):
        paths.append(str(tmp_path / f"items_{s}.jsonl"))
        ch.write_jsonl(items[s], paths[-1])
    return paths


def test_precut_read_items_and_names(tmp_path):
    import precut_htstep as pc
    v = pc.read_items(toy_items(tmp_path))
    assert set(v) == {"v_long", "v_val"} and len(v["v_long"]["items"]) == 48
    assert v["v_val"]["split"] == "val" and v["v_val"]["source"]["file"] == "part1/v_val.mp4"
    assert pc.next_shard(set()) == "clips_0000"
    assert pc.next_shard({"clips_0000", "clips_0001", "clips_0003"}, 1) == "clips_0002"     # never reuse a name


def test_precut_shard_flush(tmp_path):
    import precut_htstep as pc
    for vid in ("a", "b"):
        os.makedirs(tmp_path / "clips" / vid)
        (tmp_path / "clips" / vid / "0.npz").write_bytes(b"x" * 10)
    calls = []

    def upload(path, name):
        assert os.path.exists(path)
        if name.endswith(".manifest.jsonl"):                     # manifest only after its tar
            assert calls and calls[-1][1] == name.replace(".manifest.jsonl", ".tar")
            assert [json.loads(l)["id"] for l in open(path)] == ["a#0", "b#0", "c#0"]
        calls.append((path, name))

    sh = pc.Shard(str(tmp_path), {"clips_0000"}, upload)
    ok = lambda v: {"id": f"{v}#0", "status": "ok", "clip": f"clips/{v}/0.npz", "sha256": "s", "bytes": 10}  # noqa
    sh.add({"video_id": "a", "steps": [ok("a")]})
    sh.add({"video_id": "b", "steps": [ok("b")]})
    sh.add({"video_id": "c", "steps": [{"id": "c#0", "status": "empty_video_track"}]})
    assert sh.bytes == 20
    name, rows, vids = sh.flush()
    assert name == "clips_0001" and [c[1] for c in calls] == ["shards/clips_0001.tar", "shards/clips_0001.manifest.jsonl"]
    assert {r["shard"] for r in rows} == {"clips_0001"} and len(vids) == 3
    assert not os.path.exists(tmp_path / "clips" / "a") and not os.listdir(tmp_path / "shards")   # local copies gone
    assert sh.flush() is None and sh.bytes == 0


def test_clip_repo_check():
    import precut_htstep as pc

    class Api:
        def __init__(self, private=True, role="write", exists=True):
            self.private, self.role, self.exists = private, role, exists

        def whoami(self):
            return {"name": "me", "auth": {"accessToken": {"role": self.role}}}

        def repo_info(self, repo, repo_type):
            if not self.exists:
                raise FileNotFoundError(repo)
            return SimpleNamespace(private=self.private)

    pc.check_clip_repo("me/clips", Api())
    pc.check_clip_repo("me/clips", Api(role="fineGrained"))
    for repo, api in (("me/clips", Api(private=False)), ("me/clips", Api(role="read")), ("me/clips", Api(exists=False)),
                      ("nguyenminh04/clips", Api())):
        with pytest.raises(SystemExit):
            pc.check_clip_repo(repo, api)


def test_precut_index_rows():
    import precut_htstep as pc
    items = {"a#0": {"video_id": "a", "split": "train"}, "a#1": {"video_id": "a", "split": "train"},
             "b#0": {"video_id": "b", "split": "val"}, "c#0": {"video_id": "c", "split": "train"}}
    m = {"clips_0000": [{"id": "a#0", "status": "ok", "sha256": "s", "split": "train"},
                        {"id": "a#1", "status": "past_end", "split": "train"}]}
    rows, problems = pc.index_rows(items, m, failed={"b"})
    assert [(r["id"], r["status"]) for r in rows] == [("a#0", "ok"), ("a#1", "past_end"), ("b#0", "fetch_error"),
                                                       ("c#0", "missing")]
    assert problems == ["c#0 not cut yet"]
    m["clips_0001"] = [{"id": "a#0", "status": "ok", "sha256": "s"}, {"id": "z#0", "status": "ok"}]
    _, problems = pc.index_rows(items, m, failed={"b", "c"})
    assert any("two shards" in p for p in problems) and any("z#0" in p and "not an item" in p for p in problems)


def test_cut_video_and_finalize(mp4, image_processor, tmp_path, monkeypatch):
    import shutil
    import precut_htstep as pc
    monkeypatch.setattr(pc.io, "hf_get", lambda repo, path, out: shutil.copy(mp4, out))
    monkeypatch.setattr(pc.io, "probe", lambda p: {"codec": "mpeg4", "frames": 30, "fps": "10/1", "duration": 3.0})
    pc._IP = image_processor
    it = lambda i, s: {"id": f"v#{i}", "video_id": "v", "step_idx": i, "global_step_index": 1,  # noqa: E731
                       "partial": "Full Match", "segment": s, "text": "A: b"}
    v = {"split": "val", "source": {"repo": "r", "file": "part1/v.mp4"},
         "items": [it(0, [0.6, 2.4]), it(1, [2.0, 99.0]), it(2, [50.0, 60.0])]}
    a = SimpleNamespace(work=str(tmp_path), frames=8, pixels=50176, audio_sec=8.0)
    r = pc.cut_video("v", v, None, a)
    assert r["status"] == "ok" and not r["transcoded"] and not os.path.exists(tmp_path / "tmp" / "v")
    assert [s["status"] for s in r["steps"]] == ["ok", "ok", "past_end"]
    s0, s1 = r["steps"][:2]
    assert s0["sha256"] == pc.io.file_sha256(str(tmp_path / s0["clip"])) and s1["window"] == [2.0, pytest.approx(3.0)]
    # finalize: only ok items, window / src_transcoded from the precut index
    ch.write_jsonl(v["items"], str(tmp_path / "items_val.jsonl"))
    ch.write_jsonl([], str(tmp_path / "items_train.jsonl"))
    ch.write_jsonl(r["steps"], str(tmp_path / "precut_index.jsonl"))
    ch.cmd_finalize(SimpleNamespace(precut_index=str(tmp_path / "precut_index.jsonl"), out_dir=str(tmp_path)))
    recs = [json.loads(l) for l in open(tmp_path / "val_omni_htstep.jsonl", encoding="utf-8")]
    assert [x["id"] for x in recs] == ["v#0", "v#1"] and recs[1]["timestamps"] == s1["window"]
    assert recs[0]["clip"] == s0["clip"] and recs[0]["src_transcoded"] is False
    assert json.load(open(tmp_path / "dropped_steps.json"))["steps"] == {"v#2": "past_end"}


# ------------------------------------------------------------------------------------------------- probe + eval callback
def test_probe_plan():
    import probe_htstep as pr
    res = {1: {"status": "ok", "sec_per_step": 36.0, "max_reserved_gb": 30.0},
           2: {"status": "ok", "sec_per_step": 30.0, "max_reserved_gb": 40.0},
           4: {"status": "ok", "sec_per_step": 25.0, "max_reserved_gb": 75.0},     # >= 90 % of 80 GB
           8: {"status": "OOM", "max_reserved_gb": 79.0}}
    p = pr.plan(res, 80.0, 32 * 100, 1000, load_ms=10, val_ms=500)
    assert p["grad_cache_chunk"] == 2 and p["optimizer_steps_per_epoch"] == 100
    assert p["eval_every_frac"] == 0.5                  # val 0.142 h > 15 % of an epoch of 0.833 h
    assert pr.plan(res, 80.0, 32 * 100, 100, load_ms=10, val_ms=500)["eval_every_frac"] == 0.25
    assert pr.plan(res, 80.0, 32 * 100, 100, load_ms=5000, val_ms=500)["sec_per_step_with_loader"] == 40.0
    assert pr.plan({8: res[8]}, 80.0, 3200, 100, 10, 500)["grad_cache_chunk"] is None


def test_compare_paired_bootstrap():
    import compare_htstep as cmp
    rng = np.random.default_rng(1)
    n, d = 200, 16
    ids = np.array([f"v#{i}" for i in range(n)])
    unit = lambda x: x / np.linalg.norm(x, axis=1, keepdims=True)  # noqa: E731  (the npz vectors are L2-normalised)
    av = unit(rng.normal(size=(n, d)))
    pt = {"ids": ids, "text": unit(av + rng.normal(scale=0.5, size=(n, d))), "av": av}     # noisy text: low R@1
    best = {"ids": ids, "text": unit(av + rng.normal(scale=0.01, size=(n, d))), "av": av}  # near-perfect
    same = cmp.compare(pt, pt, n=200)
    assert same["t2v"]["R@1"]["diff"] == 0 and same["t2v"]["R@1"]["ci95"] == [0, 0]
    assert not same["v2t"]["R@5"]["significant"]
    up = cmp.compare(pt, best, n=200)
    assert up["t2v"]["R@1"]["best"] == 100.0 and up["t2v"]["R@1"]["significant"] and up["t2v"]["R@1"]["ci95"][0] > 0
    with pytest.raises(ValueError):
        cmp.compare(pt, {**best, "ids": ids[::-1]})


def test_eval_callback_save_stop_resume(tmp_path, monkeypatch):
    from qwenvl.train import retrieval_eval as rv
    r1 = iter([10.0, 20.0, 15.0, 18.0, 30.0])
    monkeypatch.setattr(rv, "encode", lambda model, ds: {"failed": []})
    monkeypatch.setattr(rv, "score", lambda emb: {"t2v": {"R@1": next(r1), "R@1_multi": 0.0}, "v2t": {"R@1": 0.0}})
    saved = []
    monkeypatch.setattr(rv, "save_lora", lambda model, d: saved.append(d))
    monkeypatch.setattr(rv, "save_embeddings", lambda emb, p: None)
    ctl = lambda: SimpleNamespace(should_save=False, should_training_stop=False)  # noqa: E731
    st = lambda s: SimpleNamespace(global_step=s, max_steps=100, epoch=s / 50)  # noqa: E731
    cb = rv.RetrievalEvalCallback(None, None, str(tmp_path), every=10, patience=2)
    cb.on_train_begin(None, st(0), ctl())                              # pt baseline: 10
    c = ctl()
    cb.on_step_end(None, st(10), c)                                    # 20: best
    assert c.should_save and not c.should_training_stop and cb.st["best_step"] == 10 and len(saved) == 1
    c = ctl()
    cb.on_step_end(None, st(15), c)                                    # not an eval step
    assert not c.should_save
    cb2 = rv.RetrievalEvalCallback(None, None, str(tmp_path), every=10, patience=2)   # resumed session
    c = ctl()
    cb2.on_step_end(None, st(10), c)                                   # already evaluated: not again
    assert not c.should_save and len(cb2.st["history"]) == 2
    cb2.on_step_end(None, st(20), ctl())                               # 15
    c = ctl()
    cb2.on_step_end(None, st(30), c)                                   # 18: second eval without a better R@1
    assert c.should_save and c.should_training_stop and cb2.st["best"] == 20.0
    assert json.load(open(tmp_path / "metrics.json"))["best_step"] == 10


def test_convert_youcookii_untouched():
    """convert_htstep.py does not refactor convert_youcookii.py, so the YC2 manifests are produced unchanged."""
    r = subprocess.run(["git", "-C", ROOT, "diff", "--quiet", "460257f", "--", "scripts/convert_youcookii.py", "scripts/eval_youcookii.py"])
    assert r.returncode == 0


# ------------------------------------------------------------------------------------------------- real data (section 7)
@pytest.mark.skipif(not os.path.exists(os.path.join(DATA, "items_train.jsonl")), reason="run convert_htstep split")
def test_items_against_independent_count():
    ann = json.load(open(os.path.join(HTSTEP, "annotations.json"), encoding="utf-8"))
    with open(os.path.join(HTSTEP, "taxonomy.csv"), newline="", encoding="utf-8") as f:
        tax = {int(r["global_step_index"]): r for r in csv.DictReader(f)}
    split = json.load(open(os.path.join(DATA, "htstep_split_omni.json"), encoding="utf-8"))
    tr, va = set(split["train_videos"]), set(split["val_videos"])
    assert not tr & va and "6MBctYaMU8U" not in tr | va
    assert all(ann[v]["subset"] == "train" for v in tr) and all(ann[v]["subset"] == "val_seen" for v in va)
    for name, vids in (("train", tr), ("val", va)):
        items = [json.loads(l) for l in open(os.path.join(DATA, f"items_{name}.jsonl"), encoding="utf-8")]
        n = 0
        for v in vids:                                        # independent count: same rule, written again
            k = 0
            for a in ann[v]["annotations"]:
                if a["segment"][1] - a["segment"][0] > 0 and k < 48:
                    k += 1
            n += k
        assert n == len(items)
        for it in items:
            r = tax[it["global_step_index"]]
            assert it["text"] == r["activity"] + ": " + r["headline"].strip()
