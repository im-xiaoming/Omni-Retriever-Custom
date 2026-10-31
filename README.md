# OmniRetriever (fixed inference)

A fork of the inference code of [OmniRetriever](https://arxiv.org/abs/2605.26641)
([YunzeLiu/OmniRetriever-7B](https://huggingface.co/YunzeLiu/OmniRetriever-7B)): text, video, audio or video+audio
-> one L2-normalised 3584-d vector in a shared space. Used to make the OmniRetriever step / clip vectors of UniAV.

The model is the WAVE-7B backbone (Qwen2.5-Omni + BEATs) with the OmniRetriever LoRA adapter on top.

## What the fork changes

| Change | Where |
|---|---|
| Video decoded with PyAV instead of decord (decord hangs on many clips). Frame counts differ, so do not mix vectors from the two | `data/media.py` |
| Audio falls back to PyAV when librosa cannot read the track; decoded clips are cached | `data/media.py` |
| Prompts end with `<|im_end|>` as in training (the head reads the last position) | `inference/encode.py` |
| Audio placeholder doubled for the interleaved BEATs vectors, as in training | `inference/encode.py` |
| Dual-modal encoders `encode_tv` (video+text) and `encode_at` (audio+text) | `inference/encode.py`, `models/loader.py` |
| BEATs weights checked after loading, standalone checkpoint as fallback | `models/beats_adaptor.py` |
| CLI: all modalities in one pass, batching, prefetch, resume from `OUTPUT.partial.npz` | `cli.py` |
| Patch of the WAVE-7B remote code so the AV prompt can end with a token | `scripts/patch_wave_rope.py` |

## Setup

Needs one GPU with >= 24 GB (A100 / H100) and ffmpeg.

    pip install -r requirements.txt
    export PYTHONPATH=$PWD/src:$PYTHONPATH

Weights:

    huggingface-cli download YunzeLiu/OmniRetriever-7B --local-dir adapters/omniretriever-7b

WAVE-7B (Hugging Face layout, BEATs inside its shards) from the WAVE release, then once:

    python scripts/patch_wave_rope.py /path/to/WAVE-7B

## Python API

```python
from omniretriever import OmniRetriever

model = OmniRetriever.from_pretrained(
    base_model='/path/to/WAVE-7B',
    adapter='adapters/omniretriever-7b',
    device='cuda', dtype='bfloat16',
)

t  = model.encode_text(['Make Pancakes: whisk the eggs'])  # (N, 3584)
av = model.encode_av(['clip.mp4'])                          # video + its audio track
v  = model.encode_video(['clip.mp4'])                       # video only
a  = model.encode_audio(['clip.mp4'])                       # audio only
print(av @ t.T)                                             # cosine: vectors are normalised
```

Text can be batched; media one clip per call.

## CLI

    python -m omniretriever.cli extract manifest.jsonl --base-model /path/to/WAVE-7B \
        --adapter adapters/omniretriever-7b --output emb.npz --modalities text av

One JSON record per line, any field may be missing:

    {"id": "...", "text": "...", "video": "clip.mp4", "audio": "clip.mp4"}

The npz holds one vector per `<id>__<modality>` (`text`, `video`, `audio`, `av`, `tv`, `at`).
`python -m omniretriever.cli evaluate emb.npz` scores retrieval (R@1/5/10, NDCG@10, MRR).

## Layout

```
src/omniretriever/
  cli.py                  extract / evaluate
  models/loader.py        OmniRetriever.from_pretrained, encode_*
  models/wave.py          WAVE-7B loader
  models/beats_adaptor.py BEATs check / fallback
  data/media.py           video frames and audio waveforms
  inference/encode.py     prompts and forward pass of every encoder
  evaluation/             retrieval metrics
scripts/patch_wave_rope.py
training/                 upstream training code, unchanged
```

## License

Apache-2.0, as upstream. Paper: Liu et al., *OmniRetriever: Any-to-Any Audio-Video-Text Retrieval via
Fusion-as-Teacher Distillation*, arXiv 2605.26641.
