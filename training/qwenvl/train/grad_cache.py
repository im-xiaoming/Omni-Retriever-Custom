"""Pairwise text <-> audio+video InfoNCE with GradCache and the same-text negative mask (plan OMNI_HTSTEP_LORA.md 4.2-4.3).

The loss is the formula of the pairwise branch of the model (modeling_qwen2_5_omni.py:3248-3268): L2-normalise the
AV embedding (``mllm_embeds``) and the text embedding (``text_embeds``), divide the two B x B similarity matrices by
the temperature, cross-entropy against the diagonal in both directions, mean. It is computed here, on one GPU and
without ``dist.nn.all_gather``, in float32. With ``mask_same_text`` every off-diagonal pair whose two step texts are
the same string is set to -inf in both directions before the cross-entropy.

The embeddings come from the model's own ``pred_embeds`` path (the one ``scripts/eval_youcookii.py`` scores), so the
loss sees exactly what validation ranks.

GradCache (Gao et al. 2021): forward every chunk of K items without graph to get all B embeddings, take the loss and
its gradient w.r.t. the embeddings, then forward each chunk again with graph and backpropagate that gradient. The
RNG state of every chunk (CPU and CUDA: LoRA dropout) is saved before the first forward and restored for the second,
so both forwards are the same function.
"""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F

EXTRA_KEYS = ("texts", "ids")


class RandContext:
    """Snapshot of the CPU and CUDA RNG; ``with ctx:`` replays it and leaves the outer RNG untouched."""

    def __init__(self):
        self.cpu = torch.get_rng_state()
        self.cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    def __enter__(self):
        devices = list(range(torch.cuda.device_count())) if self.cuda is not None else []
        self._fork = torch.random.fork_rng(devices=devices)
        self._fork.__enter__()
        torch.set_rng_state(self.cpu)
        if self.cuda is not None:
            torch.cuda.set_rng_state_all(self.cuda)

    def __exit__(self, *exc):
        return self._fork.__exit__(*exc)


def same_text_mask(texts) -> torch.Tensor:
    """(B, B) bool, True where i != j and texts[i] == texts[j]."""
    n = len(texts)
    m = torch.tensor([[texts[i] == texts[j] and i != j for j in range(n)] for i in range(n)], dtype=torch.bool)
    return m


def pair_loss(av: torch.Tensor, txt: torch.Tensor, temperature: float, mask: torch.Tensor | None = None):
    """(loss, v2t_loss, t2v_loss). ``mask`` (B, B) True = pair removed from the negatives."""
    av = F.normalize(av.float(), dim=-1)
    txt = F.normalize(txt.float(), dim=-1)
    v2t = av @ txt.t() / temperature
    t2v = txt @ av.t() / temperature
    if mask is not None:
        mask = mask.to(v2t.device)
        v2t = v2t.masked_fill(mask, float("-inf"))
        t2v = t2v.masked_fill(mask.t(), float("-inf"))
    labels = torch.arange(av.shape[0], device=av.device)
    v2t_loss = F.cross_entropy(v2t, labels)
    t2v_loss = F.cross_entropy(t2v, labels)
    return 0.5 * (v2t_loss + t2v_loss), v2t_loss, t2v_loss


def embed(model, batch: dict):
    """(av, text) embeddings of a collated batch through the model's pred_embeds path."""
    batch = {k: v for k, v in batch.items() if k not in EXTRA_KEYS}
    out = model(**batch, output_hidden_states=True, return_dict=True, pred_embeds=True)
    return out.mllm_embeds, out.text_embeds


def grad_cache_step(model, instances, collate, prepare, chunk: int, temperature: float, mask_same_text: bool,
                    backward=None, autocast=nullcontext, scale: float = 1.0) -> dict:
    """One GradCache pass over ``instances`` (the B items of one optimiser step); gradients are accumulated into
    the trainable parameters, the optimiser step is the caller's. ``collate`` turns a list of items into a batch,
    ``prepare`` moves it to the device, ``backward(t)`` backpropagates a scalar (default ``t.backward()``),
    ``scale`` multiplies the loss (1 / gradient accumulation)."""
    backward = backward or (lambda t: t.backward())
    chunks = [instances[i:i + chunk] for i in range(0, len(instances), chunk)]
    texts = [it.get("text") for it in instances]
    mask = same_text_mask(texts) if mask_same_text else None

    states, avs, txts = [], [], []
    with torch.no_grad():
        for c in chunks:
            states.append(RandContext())
            with autocast():
                a, t = embed(model, prepare(collate(c)))
            avs.append(a.detach().float())
            txts.append(t.detach().float())
    av = torch.cat(avs).requires_grad_(True)
    tx = torch.cat(txts).requires_grad_(True)
    loss, v2t_loss, t2v_loss = pair_loss(av, tx, temperature, mask)
    (loss * scale).backward()
    sizes = [len(c) for c in chunks]
    for c, state, g_av, g_tx in zip(chunks, states, av.grad.split(sizes), tx.grad.split(sizes)):
        with state:
            with autocast():
                a, t = embed(model, prepare(collate(c)))
            backward((a.float() * g_av).sum() + (t.float() * g_tx).sum())
    return {"loss": loss.detach(), "v2t_loss": v2t_loss.detach(), "t2v_loss": t2v_loss.detach(),
            "masked_pairs": int(mask.sum().item()) if mask is not None else 0}


def full_batch_step(model, instances, collate, prepare, temperature: float, mask_same_text: bool,
                    backward=None, autocast=nullcontext, scale: float = 1.0) -> dict:
    """The same loss with the whole batch in one forward with graph (GradCache off)."""
    backward = backward or (lambda t: t.backward())
    mask = same_text_mask([it.get("text") for it in instances]) if mask_same_text else None
    with autocast():
        a, t = embed(model, prepare(collate(instances)))
    loss, v2t_loss, t2v_loss = pair_loss(a, t, temperature, mask)
    backward(loss * scale)
    return {"loss": loss.detach(), "v2t_loss": v2t_loss.detach(), "t2v_loss": t2v_loss.detach(),
            "masked_pairs": int(mask.sum().item()) if mask is not None else 0}


class InstanceCollator:
    """Collator of the GradCache trainer: hands the raw items over; chunks are collated by the trainer."""

    def __call__(self, instances):
        return {"instances": list(instances)}
