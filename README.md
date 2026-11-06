# OmniRetriever: inference

Inference code for **OmniRetriever-7B**, a LoRA adapter on top of WAVE-7B
(Qwen2.5-Omni + BEATs). It maps text, video, audio and their combinations
into one embedding space for any-to-any retrieval.

Paper: [arXiv 2605.26641](https://arxiv.org/abs/2605.26641).
Fine-tuning and benchmark guides: [GUIDE_TRAIN_EVAL_RETRIEVAL.md](GUIDE_TRAIN_EVAL_RETRIEVAL.md),
[GUIDE_BENCHMARK.md](GUIDE_BENCHMARK.md). Training: [training/README.md](training/README.md).

---

## 1. Requirements

* Linux, Python ≥ 3.10, `ffmpeg` on the PATH
* One GPU with ≥ 24 GB for bf16 (A100 / H100). The backbone alone is ~16 GB in bf16.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install transformers==4.57.3 peft==0.20.0 av   # versions the fixes were tested with
export PYTHONPATH=$PWD/src:$PYTHONPATH
```

Do not reinstall `torch` on Colab: the preinstalled build already has CUDA.

## 2. Weights

| What | Upstream | Mirror |
| --- | --- | --- |
| WAVE-7B backbone + BEATs (~19.6 GB) | `tsinghua-ee/WAVE-7B` | `nguyenminh04/omni-model` |
| OmniRetriever-7B LoRA adapter (~0.8 GB) | `YunzeLiu/OmniRetriever-7B` | `nguyenminh04/omni-adapters` |

```bash
huggingface-cli download nguyenminh04/omni-model    --local-dir WAVE_HOME
huggingface-cli download nguyenminh04/omni-adapters --local-dir adapters
export WAVE_HOME=$PWD/WAVE_HOME
```

The loader expects this layout:

```
$WAVE_HOME/
├── WAVE-7B/                                          ← HF layout, remote code
└── BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt   ← must exist (name is checked)
adapters/omniretriever-7b/
├── adapter_config.json
└── adapter_model.safetensors
```

With the upstream `tsinghua-ee/WAVE-7B` the BEATs file is missing. Copy the
one shipped inside the model folder (the real BEATs weights are already in the
shards, only the file name is checked):

```bash
cp $WAVE_HOME/WAVE-7B/BEATs_iter3_plus_AS2M.pt \
   $WAVE_HOME/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt
```

Then patch the WAVE-7B remote code once (the `nguyenminh04/omni-model` mirror
is already patched). It fixes `get_rope_index` so the `av` path gets its
turn-end token, skips itself when already applied, and keeps the original as
`.prepatch`. Without it `encode_av` stops with an error.

```bash
python scripts/patch_wave_rope.py $WAVE_HOME/WAVE-7B
```

## 3. Python API

```python
from omniretriever import OmniRetriever

model = OmniRetriever.from_pretrained(
    base_model=f"{WAVE_HOME}/WAVE-7B",
    adapter="adapters/omniretriever-7b",   # local path or HF id
    device="cuda", dtype="bfloat16",
)

z_text  = model.encode_text("a dog barking in the rain")
z_video = model.encode_video("clip.mp4")
z_audio = model.encode_audio("clip.wav")
z_av    = model.encode_av("clip.mp4", audio_path="clip.wav")   # omit audio_path to read it from the mp4
z_tv    = model.encode_tv("clip.mp4", "a dog barking")          # video + caption
z_at    = model.encode_at("clip.wav", "a dog barking")          # audio + caption

# Cast to float32 before .numpy(): numpy has no bfloat16.
sim = (z_text.float() @ z_av.float().T)
```

Each call accepts one item or a list. Outputs are L2-normalised
(`InferenceConfig.normalize=True`), so the dot product is the cosine similarity.

## 4. Batch extraction

A manifest is JSON or JSONL, one record per line:

```json
{"id": "00000", "text": "Both women extend their arms...", "video": "benchmark/videos/6886118185460845830.mp4", "audio": "benchmark/videos/6886118185460845830.wav"}
```

```bash
python -m omniretriever.cli extract manifest.jsonl \
    --base-model $WAVE_HOME/WAVE-7B \
    --adapter    adapters/omniretriever-7b \
    --output     output/embeddings.npz \
    --device cuda --dtype bfloat16 --checkpoint-every 25 -v > /dev/null
```

The output `.npz` has one vector per key `{id}__{modality}` with modality in
`text video audio av tv at`.

| Flag | Default | Notes |
| --- | --- | --- |
| `--modalities` | all six | Each modality is a full pass. `text video audio av` cuts a third of the time but loses `a2tv tv2a v2at at2v`. |
| `--batch-size` | 1 | **Keep at 1.** The fusion head pools the last position and padding is not compensated: shorter samples in a batch get wrong vectors (cosine down to 0.89). |
| `--checkpoint-every` | 25 | Writes `OUTPUT.partial.npz`; rerunning the same command resumes from it (`--no-resume` to restart). |
| `--media-root` | – | Base dir for relative media paths in the manifest. |
| `--prefetch N` | 0 | Decode clips on N background threads. Not measured on GPU. |
| `--instruction-paths` | `audio av` | Paths that get the instruction prompt. It helps audio and hurts video. |

`> /dev/null` hides the two debug lines the WAVE-7B remote code prints per
forward pass (tens of thousands of lines on a full run). The progress bar goes
to stderr and stays visible.

Runtime on one A100: about 1–1.5 h for 3,510 records with all six modalities.
Video decoding on the CPU is the bottleneck.

## 5. Scoring

```bash
python scripts/score_embeddings.py output/embeddings.npz --output metrics.json
```

Scores every direction whose two modalities are present (`t2v v2t t2a a2t v2a
a2v t2av av2t a2tv tv2a v2at at2v`) with R@1/5/10 and MRR.
`python -m omniretriever.cli evaluate` does not read this `.npz` layout.

## 6. Differences from the upstream release

The upstream code does not run to the end, and when it does the numbers differ
from the paper. This fork changes:

| Change | Why |
| --- | --- |
| float32 before `numpy()` | numpy has no bfloat16; the default dtype crashed on the first record |
| PyAV instead of decord | decord hangs forever on some clips, inside C code |
| `<|im_end|>` at the end of every prompt | training puts it at the pooled position; AVG-all R@1 0.289 → 0.395 on 3,510 records |
| Instruction prompt on the audio path | t2a 0.110 → 0.197, a2t 0.103 → 0.187 (300 records) |
| No doubled audio token slots | matches the paper's token budget (Table S2) |
| `encode_tv`, `encode_at` | missing upstream; needed for 4 of the 12 directions |
| `patch_wave_rope.py` | lets the `av` path take the turn-end token |
| Partial saves, resume, progress bar | survives Colab disconnects |
| Record-major batching with a decode cache | `video`, `av`, `tv` share one decode (2.75× faster) |

Do not mix embeddings extracted with decord and with PyAV: they count frames
differently, so the 8 sampled frames differ.

---

## Citation

```bibtex
@article{liu2026omniretriever,
  title         = {OmniRetriever: Any-to-Any Audio-Video-Text Retrieval via Fusion-as-Teacher Distillation},
  author        = {Liu, Yunze and Wu, Chi-Hao and Zhou, Enmin and Shen, Junxiao},
  year          = {2026},
  eprint        = {2605.26641},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  doi           = {10.48550/arXiv.2605.26641}
}
```

License: Apache-2.0. Built on WAVE-7B (Qwen2.5-Omni + BEATs).
