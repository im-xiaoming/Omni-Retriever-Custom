#!/usr/bin/env bash
# HT-Step LoRA for OmniRetriever without the notebook (plan OMNI_HTSTEP_LORA.md section 5): the steps of
# notebooks/omni_htstep_lora.ipynb, logged, safe to rerun (every step skips what is already there).
#
#   RUN_MODE=precut bash scripts/run_vast.sh   # CPU is enough: cut the clips, upload them to HF_CLIPS (section 3)
#   RUN_MODE=probe  bash scripts/run_vast.sh   # VRAM / speed table (section 6), needs the clips of PROBE_UNIT
#   RUN_MODE=train  bash scripts/run_vast.sh   # resumes from the last local checkpoint if there is one
#   RUN_MODE=eval   bash scripts/run_vast.sh   # benchmark pt + best, export_step_emb pt + best
#   STAGES="install models clips run" (default): run only some of them, e.g. STAGES=models
#
# Env (defaults in brackets):
#   WORK [/workspace/htstep]   models, clips, runs        HF_TOKEN       read / write the private repos
#   HF_CLIPS                   team dataset repo of the clips (clips/*.tar, index/*.jsonl, precut_index.jsonl)
#   HF_RUN_REPO                team model repo receiving runs/<RUN_NAME>/{best,metrics.json}
#   RUN_NAME [htstep_lora]     GRAD_CACHE_CHUNK (train: from the probe)   GRADIENT_CHECKPOINTING [True]
#   EVAL_EVERY_FRAC [0.25]     PROBE_UNIT [1000]   PRECUT_UNITS [all]   PRECUT_JOBS [4]
#   BENCH_MANIFEST / BENCH_MEDIA_ROOT  OmniRetriever-Bench (eval, forgetting track)   ATTN_IMPL [sdpa]
set -euo pipefail
REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${WORK:-/workspace/htstep}"; RUN_MODE="${RUN_MODE:-probe}"; RUN_NAME="${RUN_NAME:-htstep_lora}"
STAGES=" ${STAGES:-install models clips run} "
mkdir -p "$WORK/logs"
LOG="$WORK/logs/${RUN_MODE}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "== $RUN_MODE [$STAGES] $(date) commit $(git -C "$REPO" rev-parse HEAD) log $LOG"
cd "$REPO"
has() { [[ "$STAGES" == *" $1 "* ]]; }

WAVE="$WORK/WAVE_HOME/WAVE-7B"; BEATS="$WORK/WAVE_HOME/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt"
PT="$WORK/adapters/omniretriever-7b"; CLIP_ROOT="$WORK/clip_root"; OUT="$WORK/runs/$RUN_NAME"
DATA="$REPO/data/htstep"
export WAVE_PATH="$WAVE" BEATS_PATH="$BEATS" LORA_CKPT="$PT" CLIP_ROOT PYTHONUNBUFFERED=1
export HTSTEP_TAXONOMY="$WORK/taxonomy.csv"

# ---- install: precut only needs the image processor of the training pin -----------------------------------------------
if has install && [ ! -f "$WORK/.installed_$RUN_MODE" ]; then
  if [ "$RUN_MODE" = "precut" ]; then
    pip install -q "transformers==4.51.3" decord av "huggingface_hub[cli]"
  else
    pip install -q -r training/requirements.txt && pip install -q decord av librosa soundfile einops "huggingface_hub[cli]"
  fi
  touch "$WORK/.installed_$RUN_MODE"
fi
python - "$RUN_MODE" <<'EOF'
import sys, torch, transformers
p = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
print("GPU", p.name if p else None, round(p.total_memory / 2**30, 1) if p else 0, "GB | torch", torch.__version__,
      "cuda", torch.version.cuda, "| transformers", transformers.__version__)
if sys.argv[1] != "precut" and (p is None or p.total_memory < 20 * 2**30 or not torch.cuda.is_bf16_supported()):
    sys.exit("không nạp được bf16, dùng L4 / A100")
EOF

# ---- models: WAVE-7B + BEATs, the pt adapter (sha256 against the author's), RoPE patch -------------------------------
if has models; then
  if [ "$RUN_MODE" = "precut" ]; then           # the image processor config is all precut reads
    [ -f "$WAVE/preprocessor_config.json" ] || hf download nguyenminh04/omni-model WAVE-7B/preprocessor_config.json \
        --local-dir "$WORK/WAVE_HOME" --quiet
  else
    [ -f "$WAVE/config.json" ] || hf download nguyenminh04/omni-model --local-dir "$WORK/WAVE_HOME" --quiet
    [ -f "$PT/adapter_model.safetensors" ] || hf download nguyenminh04/omni-adapters --include "omniretriever-7b/*" \
        --local-dir "$WORK/adapters" --quiet
    python - "$PT/adapter_model.safetensors" <<'EOF'
import hashlib, sys
from huggingface_hub import HfApi
h = hashlib.sha256()
with open(sys.argv[1], "rb") as f:
    for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
want = HfApi().get_paths_info("YunzeLiu/OmniRetriever-7B", ["adapter_model.safetensors"], expand=True)[0].lfs.sha256
print("CHECK adapter_is_author", "OK" if h.hexdigest() == want else "FAIL", h.hexdigest())
sys.exit(h.hexdigest() != want)
EOF
    python scripts/patch_wave_rope.py "$WAVE"
    [ -f "$HTSTEP_TAXONOMY" ] || curl -sL -o "$HTSTEP_TAXONOMY" \
        https://raw.githubusercontent.com/facebookresearch/htstep/main/data/taxonomy.csv
  fi
fi

# ---- clips: precut (CPU) or download + unpack ---------------------------------------------------------------------------
fetch_clips() {   # $1 = include pattern
  hf download "$HF_CLIPS" --repo-type dataset --include "$1" --local-dir "$WORK/clips_dl" --quiet
  mkdir -p "$CLIP_ROOT/.done"
  for t in "$WORK"/clips_dl/clips/*.tar; do
    [ -e "$t" ] || continue
    [ -f "$CLIP_ROOT/.done/$(basename "$t")" ] && continue
    tar -xf "$t" -C "$CLIP_ROOT" && touch "$CLIP_ROOT/.done/$(basename "$t")" && rm -f "$t"
  done
}
if has clips; then
  case "$RUN_MODE" in
    precut)
      python scripts/precut_htstep.py --items "$DATA/items_train.jsonl" "$DATA/items_val.jsonl" --processor "$WAVE" \
          --work "$WORK/precut" --hf-repo "$HF_CLIPS" --jobs "${PRECUT_JOBS:-4}" ${PRECUT_UNITS:+--units $PRECUT_UNITS}
      ;;
    probe)
      fetch_clips "clips/clips_${PROBE_UNIT:-1000}.tar"
      python - "$WORK/probe_mp4" "${PROBE_UNIT:-1000}" <<'EOF'
import os, sys, tarfile
from huggingface_hub import hf_hub_download
out, unit = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
if not any(f.endswith(".mp4") for f in os.listdir(out)):
    tar = hf_hub_download("nguyenminh04/htstep-videos", f"videos/htstep_videos_{unit}.tar", repo_type="dataset",
                          local_dir=os.path.join(out, "_dl"))
    with tarfile.open(tar) as tf:
        tf.extractall(out, members=[m for m in tf.getmembers() if m.name.endswith(".mp4")], filter="data")
    os.remove(tar)
print(len([f for f in os.listdir(out) if f.endswith(".mp4")]), "mp4 for the equivalence check")
EOF
      # the probe does not need committed manifests: finalize what is cut so far into WORK
      mkdir -p "$WORK/data" && cp "$DATA"/items_*.jsonl "$WORK/data/"
      hf download "$HF_CLIPS" precut_index.jsonl --repo-type dataset --local-dir "$WORK/clips_dl" --quiet
      python scripts/convert_htstep.py finalize --precut-index "$WORK/clips_dl/precut_index.jsonl" --out-dir "$WORK/data"
      ;;
    train) fetch_clips "clips/*.tar" ;;
  esac
fi
has run || exit 0

case "$RUN_MODE" in
  probe)
    python scripts/probe_htstep.py --train-manifest "$WORK/data/train_omni_htstep.jsonl" \
        --val-manifest "$WORK/data/val_omni_htstep.jsonl" --mp4-root "$WORK/probe_mp4" --out "$WORK/probe" \
        --n-train "$(wc -l < "$DATA/items_train.jsonl")" --n-val "$(wc -l < "$DATA/items_val.jsonl")"
    ;;
  train)
    python scripts/make_protocol.py check
    : "${GRAD_CACHE_CHUNK:?set GRAD_CACHE_CHUNK from the probe}"
    FRAC="${EVAL_EVERY_FRAC:-0.25}"
    EVERY=$(python -c "n=sum(1 for _ in open('$DATA/train_omni_htstep.jsonl')); print(max(1, int(-(-n//32)*$FRAC)))")
    INIT=True; ls -d "$OUT"/checkpoint-* >/dev/null 2>&1 && INIT=False
    TRACK=()
    if [ -n "${BENCH_MANIFEST:-}" ]; then
      [ -f "$WORK/track_bench1000.jsonl" ] || python scripts/eval_bench.py --manifest "$BENCH_MANIFEST" \
          --media-root "$BENCH_MEDIA_ROOT" --make-track "$WORK/track_bench1000.jsonl"
      TRACK=(--track_manifest "$WORK/track_bench1000.jsonl")
    fi
    USE_DEEPSPEED=False NUM_GPUS=1 TRAIN_SEED=42 DATA_PATH="$DATA/train_omni_htstep.jsonl" OUTPUT_DIR="$OUT" \
    LORA_ONLY=True LORA_INIT_ONLY=$INIT USE_TUPLE_INFONCE=False BATCH_SIZE=32 GRAD_ACCUM=1 EPOCHS=2 LR=1e-4 \
    GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-True}" \
    bash training/train.sh --grad_cache_chunk "$GRAD_CACHE_CHUNK" --mask_same_text True \
        --eval_manifest "$DATA/val_omni_htstep.jsonl" --eval_every_frac "$FRAC" --early_stop_patience 3 "${TRACK[@]}" \
        --hf_run_repo "${HF_RUN_REPO:-}" --run_name "$RUN_NAME" --adam_beta2 0.95 --seed 42 \
        --save_strategy steps --save_steps "$EVERY" --save_total_limit 1
    ;;
  eval)
    B="$WORK/bench"; mkdir -p "$B"
    [ -f "$B/bench_pt.json" ] || python scripts/eval_bench.py --adapter "$PT" --tag pt --out "$B"
    [ -f "$WORK/omni_step_emb_pt.npz" ] || python scripts/export_step_emb.py --adapter "$PT" --tag pt \
        --out "$WORK/omni_step_emb_pt.npz"
    if [ -d "$OUT/best" ]; then
      [ -f "$B/bench_best.json" ] || python scripts/eval_bench.py --lora "$OUT/best" --tag best --out "$B"
      [ -f "$WORK/omni_step_emb_best.npz" ] || python scripts/export_step_emb.py --lora "$OUT/best" --tag best \
          --out "$WORK/omni_step_emb_best.npz"
    fi
    ;;
esac
