# Chạy HT-Step LoRA trên Vast.ai qua SSH

Mọi bước là `scripts/run_vast.sh`, cùng các lệnh mà `notebooks/omni_htstep_lora.ipynb` gọi. Chạy lại an toàn: bước nào
xong rồi thì bỏ qua, train tự resume từ checkpoint cục bộ mới nhất.

## 1. Tạo instance

- Image PyTorch có CUDA (ví dụ `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel`), GPU theo bảng probe (≥ 24GB, Ampere trở lên vì cần bf16).
- Đĩa ≥ 100GB: model khoảng 18GB, clip cắt sẵn (xem `precut_summary.json`), checkpoint.
- Mở SSH, rồi:

```bash
ssh -p <port> root@<host>
tmux new -s htstep                       # mất kết nối SSH vẫn chạy tiếp; vào lại: tmux attach -t htstep
git clone -b htstep_adapter https://github.com/im-xiaoming/Omni-Retriever-Custom.git /workspace/Omni-fix
cd /workspace/Omni-fix
```

## 2. Biến môi trường

```bash
export HF_TOKEN=hf_...                         # đọc htstep-videos / omni-model, ghi repo của nhóm
export WORK=/workspace/htstep
export HF_CLIPS=<nhóm>/htstep-omni-clips       # repo dataset riêng (private) chứa clip
export HF_RUN_REPO=<nhóm>/htstep-omni-runs     # repo model riêng (private) nhận checkpoint tốt nhất + metrics.json
export GRAD_CACHE_CHUNK=<K từ bảng probe>  GRADIENT_CHECKPOINTING=True  EVAL_EVERY_FRAC=0.25
export BENCH_MANIFEST=/workspace/bench/manifests/omniretriever_bench_final.jsonl  BENCH_MEDIA_ROOT=/workspace
```

`EVAL_EVERY_FRAC` lấy theo luật của protocol: 0.5 nếu một lần eval val tốn hơn 15 % một epoch trong bảng probe, ngược lại 0.25.

## 3. Chạy

```bash
RUN_MODE=train bash scripts/run_vast.sh        # cài, tải model + adapter (kiểm sha256), tải clip, kiểm protocol, train
RUN_MODE=eval  bash scripts/run_vast.sh        # benchmark pt + checkpoint tốt nhất, vector bước cho UniAV
```

Lượt train dừng ngay nếu `make_protocol.py check` báo manifest khác protocol đã commit.

## 4. Theo dõi

```bash
tail -f $WORK/logs/train_*.log | grep -E "\[gc\]|\[eval\]|\[track\]|Error|OOM"
cat $WORK/runs/htstep_lora/metrics.json
nvidia-smi
```

Sau mỗi lần eval, `runs/<RUN_NAME>/best` (chỉ tensor LoRA) và `metrics.json` được upload lên `HF_RUN_REPO`, nên máy bị
ngắt cũng không mất checkpoint tốt nhất.

## 5. Tải kết quả về

```bash
hf download $HF_RUN_REPO --include "runs/htstep_lora/*" --local-dir D:/Học/KL/Result/omni_htstep_lora
scp -P <port> -r root@<host>:/workspace/htstep/bench D:/Học/KL/Result/omni_htstep_lora/
scp -P <port> root@<host>:/workspace/htstep/omni_step_emb_*.npz D:/Học/KL/Result/omni_htstep_lora/
```

Checkpoint `best` chỉ có LoRA; muốn có adapter đầy đủ (head của bản pt):
`python scripts/materialize_adapter.py <adapter pt> runs/htstep_lora/best <thư mục ra>`.
