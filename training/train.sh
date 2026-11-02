#!/usr/bin/env bash
#
# OmniRetriever-7B training launcher.
#
# Required env:
#   WAVE_PATH        path to the WAVE-7B backbone (HuggingFace layout)
#   BEATS_PATH       path to the BEATs audio-encoder checkpoint (.pt)
#   DATA_PATH        path to the training manifest (.jsonl, one record per line)
#
# Optional env (sensible defaults):
#   OUTPUT_DIR       checkpoint output dir              (default: ./output/omniretriever_7b)
#   NUM_GPUS         GPUs to use                        (default: 4)
#   MASTER_PORT      deepspeed master port              (default: 29503)
#   EPOCHS           training epochs                    (default: 1)
#   BATCH_SIZE       per-device micro-batch             (default: 8)
#   GRAD_ACCUM       gradient accumulation steps        (default: 8)
#   LR               learning rate                      (default: 1e-5)
#   LORA_R           LoRA rank                          (default: 16)
#   LORA_ALPHA       LoRA alpha                         (default: 32). Both are ignored when
#                    LORA_CKPT is set: the adapter's own adapter_config.json wins.
#   LORA_ONLY        what is trained on top of LORA_CKPT (default: False):
#                      False -> LoRA (q/k/v of all 28 LLM layers) + classify_linear
#                               + beats_ln + beats_proj
#                      True  -> LoRA only; the three heads keep the adapter's weights
#                    Everything else (LLM, ViT, Whisper, BEATs encoder) is always frozen.
#   VIDEO_BLACKLIST  optional path to a one-id-per-line blacklist file
#
# Media root dirs (optional – only needed when JSONL contains bare filenames):
#   VIDEO_ROOT       root directory for video files  (e.g. /data/videos)
#   AUDIO_ROOT       root directory for audio files  (e.g. /data/audio). Only records with an
#                    "audio" field use it; video-only records cut the event audio out of the video.
#   IMAGE_ROOT       root directory for image files  (e.g. /data/images)
#
# Example:
#   WAVE_PATH=/data/WAVE-7B \
#   BEATS_PATH=/data/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt \
#   DATA_PATH=/data/omniretriever_1m.jsonl \
#   bash training/train.sh
#
# Loss (USE_TUPLE_INFONCE picks exactly one of the two; they never run together):
#   True  -> Tuple-InfoNCE + fusion-as-teacher (the paper's pre-training recipe).
#            The anchor is audio+video+CAPTION; text, audio-only and video-only are
#            pulled towards that detached anchor and towards each other. The
#            audio+video embedding that retrieval actually ranks (caption absent) is
#            never in the loss -- it only moves through the shared LoRA.
#   False -> pairwise symmetric InfoNCE between the audio+video embedding and the
#            caption embedding: exactly the pair eval_youcookii.py scores. Use this
#            to fine-tune for text <-> event-clip retrieval (YouCookII / UEMR).
#   USE_TUPLE_INFONCE=False is also ~3x cheaper per step (2 forwards instead of ~6),
#   which leaves room for a larger BATCH_SIZE, i.e. more in-batch negatives.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

# --- default paths if not set by environment ---
WAVE_PATH="${WAVE_PATH:-$(cd -- "${REPO_ROOT}/../../WAVE_HOME/WAVE-7B" 2>/dev/null && pwd || echo "D:/Học/KL/Code/Omni/WAVE_HOME/WAVE-7B")}"
BEATS_PATH="${BEATS_PATH:-$(cd -- "${REPO_ROOT}/../../WAVE_HOME" 2>/dev/null && pwd || echo "D:/Học/KL/Code/Omni/WAVE_HOME")/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt2.pt}"
DATA_PATH="${DATA_PATH:-$(cd -- "${REPO_ROOT}/../../../../Data/YouCookII/metadata" 2>/dev/null && pwd || echo "D:/Học/KL/Data/YouCookII/metadata")/train_omni_video.jsonl}"
VIDEO_ROOT="${VIDEO_ROOT:-$(cd -- "${REPO_ROOT}/../../../../Data/YouCookII/videos" 2>/dev/null && pwd || echo "D:/Học/KL/Data/YouCookII/videos")}"
AUDIO_ROOT="${AUDIO_ROOT:-$(cd -- "${REPO_ROOT}/../../../../Data/YouCookII/audio" 2>/dev/null && pwd || echo "D:/Học/KL/Data/YouCookII/audio")}"
LORA_CKPT="${LORA_CKPT:-$(cd -- "${REPO_ROOT}/../../adapters/omniretriever-7b" 2>/dev/null && pwd || echo "D:/Học/KL/Code/Omni/adapters/omniretriever-7b")}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/output/omniretriever_7b}"

export BEATS_PATH
export VIDEO_ROOT
export AUDIO_ROOT
export IMAGE_ROOT="${IMAGE_ROOT:-}"

# --- auto-detect GPU count if not set ---
if [ -z "${NUM_GPUS:-}" ]; then
  if command -v nvidia-smi &>/dev/null; then
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
  elif command -v python3 &>/dev/null; then
    NUM_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
  else
    NUM_GPUS=1
  fi
  NUM_GPUS=$(( NUM_GPUS > 0 ? NUM_GPUS : 1 ))
fi

# --- optional knobs ---
MASTER_PORT="${MASTER_PORT:-29503}"
EPOCHS="${EPOCHS:-1}"
if [ "${NUM_GPUS}" -le 1 ]; then
  BATCH_SIZE="${BATCH_SIZE:-1}"
else
  BATCH_SIZE="${BATCH_SIZE:-8}"
fi
GRAD_ACCUM="${GRAD_ACCUM:-8}"
LR="${LR:-1e-5}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_INIT_ONLY="${LORA_INIT_ONLY:-True}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-True}"
USE_TUPLE_INFONCE="${USE_TUPLE_INFONCE:-True}"
LORA_ONLY="${LORA_ONLY:-False}"

echo "============================================================"
echo "  OmniRetriever-7B Training Launcher (Bash)"
echo "============================================================"
echo "GPU COUNT  : ${NUM_GPUS}"
echo "WAVE_PATH  : ${WAVE_PATH}"
echo "BEATS_PATH : ${BEATS_PATH}"
echo "DATA_PATH  : ${DATA_PATH}"
echo "VIDEO_ROOT : ${VIDEO_ROOT}"
echo "AUDIO_ROOT : ${AUDIO_ROOT}"
echo "LORA_CKPT  : ${LORA_CKPT}"
echo "OUTPUT_DIR : ${OUTPUT_DIR}"
echo "BATCH_SIZE : ${BATCH_SIZE} | GRAD_ACCUM: ${GRAD_ACCUM} | EPOCHS: ${EPOCHS}"
echo "============================================================"

# Kiểm tra đường dẫn tồn tại
if [ ! -e "${WAVE_PATH}" ]; then
  echo "LỖI: Không tìm thấy thư mục WAVE_PATH: ${WAVE_PATH}"
  echo "Vui lòng export đúng đường dẫn thực tế trên môi trường của bạn (ví dụ: export WAVE_PATH=/content/WAVE-7B)"
  exit 1
fi
if [ ! -e "${BEATS_PATH}" ]; then
  echo "LỖI: Không tìm thấy file BEATS_PATH: ${BEATS_PATH}"
  exit 1
fi
if [ ! -e "${DATA_PATH}" ]; then
  echo "LỖI: Không tìm thấy file DATA_PATH: ${DATA_PATH}"
  exit 1
fi

# USE_DEEPSPEED=False forces the plain python launch (one GPU; needed by --grad_cache_chunk, whose several backward
# passes per optimiser step DeepSpeed's engine does not expect).
if [ "${USE_DEEPSPEED:-auto}" != "False" ] && command -v deepspeed &>/dev/null; then
  echo "Chạy với DeepSpeed (GPUs: ${NUM_GPUS}, Port: ${MASTER_PORT})..."
  deepspeed --num_gpus="${NUM_GPUS}" --master_port="${MASTER_PORT}" \
    "${REPO_ROOT}/qwenvl/train/train_qwen.py" \
    --deepspeed "${REPO_ROOT}/configs/ds_zero0.json" \
    --model_name_or_path "${WAVE_PATH}" \
    --model_base         "${WAVE_PATH}" \
    --dataset_use        "${DATA_PATH}" \
    --bf16 True \
    --output_dir         "${OUTPUT_DIR}" \
    --num_train_epochs   "${EPOCHS}" \
    --per_device_train_batch_size "${BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --learning_rate      "${LR}" \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --model_max_length 2048 \
    --dataloader_num_workers 4 \
    --train_classify True \
    --classify_type all_layer \
    --pred_embeds True \
    --use_beats True \
    --tune_beats_proj True \
    --fixed_audio_duration 8 \
    --video_max_frames 8 \
    --video_min_frames 8 \
    --max_pixels 50176 \
    --min_pixels 50176 \
    --use_lora True \
    --lora_r     "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --use_tuple_infonce  "${USE_TUPLE_INFONCE}" \
    --lora_ckpt "${LORA_CKPT}" \
    --lora_init_only "${LORA_INIT_ONLY}" \
    --lora_only      "${LORA_ONLY}" \
    --gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 5 \
    --report_to none \
    --disable_tqdm False \
    "$@"
else
  echo "DeepSpeed không được tìm thấy, chạy trực tiếp bằng Python..."
  python -u "${REPO_ROOT}/qwenvl/train/train_qwen.py" \
    --model_name_or_path "${WAVE_PATH}" \
    --model_base         "${WAVE_PATH}" \
    --dataset_use        "${DATA_PATH}" \
    --bf16 True \
    --output_dir         "${OUTPUT_DIR}" \
    --num_train_epochs   "${EPOCHS}" \
    --per_device_train_batch_size "${BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRAD_ACCUM}" \
    --learning_rate      "${LR}" \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --model_max_length 2048 \
    --dataloader_num_workers 4 \
    --train_classify True \
    --classify_type all_layer \
    --pred_embeds True \
    --use_beats True \
    --tune_beats_proj True \
    --fixed_audio_duration 8 \
    --video_max_frames 8 \
    --video_min_frames 8 \
    --max_pixels 50176 \
    --min_pixels 50176 \
    --use_lora True \
    --lora_r     "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --use_tuple_infonce  "${USE_TUPLE_INFONCE}" \
    --lora_ckpt "${LORA_CKPT}" \
    --lora_init_only "${LORA_INIT_ONLY}" \
    --lora_only      "${LORA_ONLY}" \
    --gradient_checkpointing "${GRADIENT_CHECKPOINTING}" \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 5 \
    --report_to none \
    --disable_tqdm False \
    "$@"
fi
exit $?
