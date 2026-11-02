# Nhật ký: chạy thử train LoRA Omni trên tập HT-Step rất nhỏ (Colab Pro, GPU)

Mục tiêu:
1. Bắt lỗi trước lượt train thật.
2. Đo VRAM khi load model và khi train.

Lượt này không dùng precut, GradCache, mask negative hay script Vast. Code của các phần đó vẫn nằm trên nhánh, nhưng
mặc định tắt (`grad_cache_chunk=0`, không có `eval_manifest`, record không có trường `clip`).

## 2026-11-02 / 03: agent, lượt smoke (local; phần GPU chờ user chạy)

**Nhánh:** `htstep_lora`. Bản trên remote `origin/htstep_lora` = a4f0ae1; local thêm 96447c7 (TAI_VIDEO) và commit của
lượt này. Chưa push. Yêu cầu viết `htstep-lora`, còn nhánh trên GitHub tên `htstep_lora`, nên notebook clone `htstep_lora`.

### Đã sửa

| File | Chỗ | Việc |
| --- | --- | --- |
| `scripts/convert_htstep.py` | `:15-19` (docstring), `:300-334` (`decodable`, `smoke_window`, `smoke_select`), `:336-370` (`cmd_smoke`), `:374-383` (CLI) | Lệnh mới `smoke`, chi tiết ở dưới bảng |
| `training/qwenvl/train/train_qwen.py` | `:107-143` (`VramCallback`), `:510-511` (đăng ký) | Đo VRAM và thời gian mỗi bước, chi tiết ở dưới bảng |
| `scripts/smoke_eval.py` | mới | Nạp WAVE-7B + BEATs + checkpoint (`htstep_common.load_model`), encode `--n` bước đầu ở batch 1 bằng `retrieval_eval.encode` (một forward ra cả text lẫn audio-video, đường `pred_embeds`). Ghi `after_load_gb`, đỉnh VRAM lúc eval, ms / item, t2v / v2t của các item đó |
| `tests/test_htstep_lora.py` | `:81-98` (`test_smoke_subset`) | Kiểm luật lọc codec, chỉ lấy video của tar train, val chỉ lấy nguồn tar, kẹp `end`, `past_end`, record không có `clip` |
| `notebooks/omni_htstep_smoke.ipynb` | ô `# ===== 0` → `6` | Notebook GPU, mô tả ở mục "Lệnh cho user" |
| `data/htstep/smoke/smoke_{train,val}.jsonl` | mới | Manifest nhỏ đã sinh ở local, chi tiết ở dưới bảng |

**Lệnh `smoke` của `convert_htstep.py`:**
- Lấy bước từ `items_{train,val}.jsonl`, tức là đã đúng luật 2.2 / 2.4: giữ Full + Partial; `end > start`; 48 bước đầu; text `f"{activity}: {headline.strip()}"` lấy từ taxonomy; id `f"{video_id}#{step_idx}"`.
- `--limit-videos 30 5` chọn 30 video train của `htstep_videos_4007.tar` và 5 video val có nguồn tar, theo thứ tự id.
- Chỉ lấy h264 / vp9 có `pix_fmt` theo `docs/htstep/tai_video/codecs.jsonl`, nên AV1 và track rỗng bị bỏ.
- Cửa sổ kẹp theo duration của ffprobe; bước có `start ≥ duration` thì bỏ (`past_end`).
- Record dùng `build_record` của `finalize`, bỏ `clip`.
- `--video-root` tải đúng các mp4 đó bằng `range_get` của `scripts/bench_precut_one_tar.py`.

**`VramCallback`** ghi vào `<OUTPUT_DIR>/vram.json`:
- `torch.cuda.max_memory_allocated` / `max_memory_reserved` (GB) ở `on_train_begin`, lúc model + adapter đã lên GPU nhưng chưa có state của optimizer (`after_load`);
- giá trị đó ở mỗi lần log (`logging_steps 1`), cùng loss;
- wall time mỗi bước optimizer.

File được ghi lại sau mỗi lần log, nên lượt bị dừng vì OOM vẫn giữ được số đã đo. Callback chỉ đọc bộ đếm, không đổi hành vi train.

**Manifest nhỏ:**
- `smoke_train.jsonl`: 210 bước / 30 video, sha256 `f0388e20…75e0ec`.
- `smoke_val.jsonl`: 29 bước / 5 video, sha256 `5dfbfa48…114421`.
- Không có bước `past_end`; 2 bước train bị kẹp `end`.

### Đã chạy ở local

| Lệnh | Kết quả |
| --- | --- |
| `python scripts/convert_htstep.py smoke --limit-videos 30 5` | 2 manifest như trên |
| `convert_htstep.py smoke --limit-videos 1 1 --out-dir <scratch> --video-root <scratch>/videos` | Tải range 2 mp4 (`5mOfGfp7yFc` 6.4 MB, `-25-1nbOki0` 36.6 MB), đủ byte |
| `python -m pytest tests/test_htstep_lora.py -q` (transformers 4.51.3) | **12 passed** (11 cũ + `test_smoke_subset`) |
| Kiểm cú pháp 7 ô của notebook; regex đọc báo cáo tham số trên đúng định dạng in của `train_qwen.py:475` | OK, đọc ra `[('LoRA', '6,881,280')]` |

Chưa chạy được ở local: mọi thứ cần GPU (train, `VramCallback`, `smoke_eval.py`).

### Lệnh cho user (Colab Pro, runtime L4 hoặc A100)

1. Push nhánh: `git -C D:\Học\KL\Code\Omni\Omni-fix push origin htstep_lora`. Agent không push.
2. Mở `notebooks/omni_htstep_smoke.ipynb` trên Colab, chọn GPU L4 hoặc A100. (Tuỳ chọn) đặt secret `HF_TOKEN`: các repo đều public, token chỉ để tránh rate limit.
3. Chạy ô 0 → 6:

| Ô | Việc |
| --- | --- |
| 1 | Clone `htstep_lora` (in commit). Cài `training/requirements.txt` + decord av. Đọc GPU bằng `nvidia-smi` (kernel không tạo CUDA context, để không giữ VRAM). Compute < 8.0 (T4) thì dừng với câu "T4 không chạy được bf16, đổi sang L4 hoặc A100" |
| 2 | Tải `nguyenminh04/omni-model` và `nguyenminh04/omni-adapters/omniretriever-7b`, `CHECK adapter_is_author` (sha256 so với `YunzeLiu/OmniRetriever-7B`), chạy `patch_wave_rope.py` |
| 3 | `convert_htstep.py smoke --video-root …` (35 mp4, khoảng 0.7 GB). Kiểm manifest trùng bản đã commit (`CHECK smoke_manifest_committed OK`) |
| 4 | Với mỗi batch 1 / 2 / 4 / 8, chạy một tiến trình `training/train.sh` riêng, chi tiết ở dưới bảng |
| 5 | `smoke_eval.py` trên 20 bước val, batch 1, với checkpoint của batch lớn nhất chạy được |
| 6 | In bảng `GPU \| VRAM tổng \| sau khi load \| đỉnh train bs1/2/4/8 (hoặc OOM) \| giây / bước \| đỉnh eval \| lỗi`, ghi `vram_smoke_<GPU>.json`, chép file đó + log lên Drive (`DRIVE_OUT`) |

**Ô 4 (train thử):**
- Biến môi trường: `LORA_CKPT` = adapter gốc, `LORA_ONLY=True`, `LORA_INIT_ONLY=True`, `USE_TUPLE_INFONCE=False`, `GRADIENT_CHECKPOINTING=True`, `GRAD_ACCUM=1`, `NUM_GPUS=1`, `TRAIN_SEED=42`, `USE_DEEPSPEED=False`.
- Tham số thêm: `--max_steps 20 --save_steps 20 --save_total_limit 1`, bf16 (đã có sẵn trong `train.sh`).
- Kiểm khối "Trainable parameters by group" chỉ có đúng một dòng `LoRA 6,881,280`.
- OOM được ghi `"OOM"` rồi chạy batch tiếp, ô không dừng. `QwenVLTrainer` tự bỏ qua batch bị OOM (`trainer.py:132-140`) và chỉ dừng khi OOM 3 lần liên tiếp, nên ô 4 đếm cả dòng `[OOM] Skipping`. Một batch bị bỏ cũng tính là OOM.
- Số dòng `[ERROR] Sample` (mp4 lỗi, loader thay bằng mẫu ngẫu nhiên) được ghi vào cột lỗi.

Gửi lại cho agent: `vram_smoke_<GPU>.json`, cùng `train_bs*.log` / `eval.log` nếu có lỗi.

### Ghi chú

- `USE_DEEPSPEED=False`: chạy Python thường trên 1 GPU, đúng như lượt train thật với GradCache sau này (`train.sh:130-132`). `after_load` vì thế là model đã lên GPU qua `Trainer.__init__`.
- `after_load` đo ở `on_train_begin`, nên **chưa** gồm state AdamW. State này tạo ở bước đầu và có trong đỉnh khi train.
- Đỉnh khi train là `max_memory_allocated` cộng dồn qua 20 bước. Giây / bước là trung vị, bỏ 2 bước đầu.
- `dataloader_num_workers 4` (mặc định của `train.sh`): mp4 được giải mã trên CPU ngay lúc train, nên giây / bước có thể bị giới hạn bởi CPU của Colab chứ không phải GPU.
