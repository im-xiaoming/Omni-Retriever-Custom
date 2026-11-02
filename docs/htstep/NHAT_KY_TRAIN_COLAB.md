# Nhật ký: train LoRA Omni trên HT-Step bằng Colab (khuôn YC2)

Khuôn: `GUIDE_TRAIN_EVAL_RETRIEVAL.md` mục C (Drive, clone, cài thư viện, tải trọng số, biến môi trường,
`training/train.sh`, eval). Không cắt clip trước, không dùng GradCache, không che negative trùng text. Các phần đã commit
trước đây (precut, smoke, Vast) vẫn nằm trên nhánh nhưng lượt này không dùng.

## 2026-11-03: agent (local + Colab CPU; phần GPU chờ user)

**Nhánh:** `htstep_lora`. Yêu cầu viết `htstep-lora`, còn nhánh trên GitHub tên `htstep_lora`. Chưa push.

### Đã sửa

| File | Chỗ | Việc |
| --- | --- | --- |
| `scripts/download_htstep.py` | mới (190 dòng) | Tải mp4 vào `--video-root/<id>.mp4`, chi tiết ở dưới bảng |
| `scripts/convert_htstep.py` | `:21-26` (docstring), `:344-352` `decord_window`, `:354-369` `manifest_records`, `:372-393` `cmd_manifest`, `:437-442` CLI | Lệnh `manifest`, chi tiết ở dưới bảng |
| `training/qwenvl/data/data_qwen.py` | `:454-455` | Ngưỡng bỏ mp4 lớn: `500 MB` → `MAX_VIDEO_MB` (mặc định vẫn 500, YC2 không đổi). Lý do ở dưới bảng |
| `training/qwenvl/train/train_qwen.py` | `:107-144` `VramCallback`, `:510-511` đăng ký | `max_memory_allocated` / `max_memory_reserved` (GB) ngay khi model + adapter lên GPU (`on_train_begin`), ở mỗi bước log, wall time mỗi bước. Ghi `<OUTPUT_DIR>/vram.json` sau mỗi lần log; cuối lượt in dòng `[vram] …`. Chỉ đọc bộ đếm, không đổi hành vi train. Callback có từ commit 54d0849; lượt này thêm phần in ở `:144` |
| `scripts/eval_youcookii.py` | `:82` `--name`, `:176-181` VRAM sau load + reset đỉnh, `:208` giờ bắt đầu, `:275-280` thêm `ndcg@10_pct`, `vram_gb`, `ms_per_item`, `:290` dòng NDCG@10, `:294-295` dòng VRAM | Chấm manifest HT-Step bằng chính script YC2, chi tiết ở dưới bảng |
| `tests/test_htstep_lora.py` | `:102` `test_decord_window_matches_video_decord`, `:111` `test_manifest_from_download_log`, `:123` `test_download_pick_has_av1` | Test cho các phần trên |
| `notebooks/omni_htstep_train.ipynb` | ô `# ===== 1` → `8` | Notebook Colab, mô tả ở mục "Cách chạy notebook" |

**`download_htstep.py`:**
- Train: range read đúng member trong tar (offset từ `data/htstep/tar_index.csv`, dùng `range_get`).
- Val: file của `htstep-val-seen-videos`. 162 video có hai bản thì split đã chọn bản val repo.
- ffprobe từng file:
  - AV1 thì chuyển mã bằng đúng lệnh `ffmpeg … -c:v libx264 -qp 0 -preset ultrafast -pix_fmt yuv420p -fps_mode passthrough -c:a copy`. Số khung, fps và độ dài trước / sau phải bằng nhau (lệch ≤ 1 khung); khác thì bỏ video (`transcode_mismatch`).
  - Không có khung video thì bỏ (`empty_video_track`).
- Cuối cùng decord mở file, đọc thử một khung, và ghi số khung / fps / độ dài vào log.
- `--n-train-videos 20 --n-val-videos 5` (0 = cả split): lấy n video đầu theo id trong số video đọc được của `codecs.jsonl`; tập train luôn có 1 video AV1.
- Resume theo `download_log.jsonl`: bỏ qua video đã xong; file raw đúng size thì không tải lại. Lỗi mạng được retry (backoff).

**Lệnh `manifest` của `convert_htstep.py`:**
- Bước lấy từ `items_*.jsonl`, nên đúng luật UniAV 2.2 và text / id 2.4.
- Chỉ lấy video `ok` trong download log.
- Cửa sổ kiểm bằng **đúng điều kiện của `video_decord`** (`data_qwen.py:480-485`), tính trên số khung / fps mà decord đọc được: kẹp `end`, bỏ `past_end`.
- Record theo định dạng YC2: `conversations`, `video`, `timestamps`, `text` + các trường HT-Step, không có `clip`.
- Ra `train_omni_htstep.jsonl`, `val_omni_htstep.jsonl`, `manifest_summary.json` (số video / bước bị loại theo lý do).

**Vì sao cần `MAX_VIDEO_MB`:** bản H.264 lossless của video AV1 dài 975 s nặng 1.42 GB, nên loader báo `Video too large (1417MB), skipping` và thay bằng mẫu ngẫu nhiên. Notebook đặt `MAX_VIDEO_MB=4096`.

**`eval_youcookii.py`:**
- Thêm `--name` (mặc định `YouCookII`).
- Ghi VRAM sau khi load, VRAM đỉnh khi eval (allocated / reserved) và ms / item.
- In thêm dòng NDCG@10 (%).
- Các khoá cũ của JSON kết quả không đổi.
- Script vẫn chấm batch 1, N → N theo đường chéo, T→AV và AV→T, bằng `all_metrics`.

### Đã chạy

| Ở đâu | Lệnh | Kết quả |
| --- | --- | --- |
| local | `pytest tests/test_htstep_lora.py -q` (transformers 4.51.3) | **15 passed** |
| Colab CPU (runtime tạm qua `colab` CLI, đã tắt) | `download_htstep.py --n-train-videos 3 --n-val-videos 2` | 5/5 `ok`, gồm 1 AV1 (`--SjTRJyp4w`) đã chuyển mã: 23 389 khung, 24000/1001 fps, 975.52 s trước và sau. 1 m 57 s; riêng bước chuyển mã mất 116 s trên 2 vCPU |
| Colab CPU | chạy lại cùng lệnh | `5 already done` (resume) |
| Colab CPU | `convert_htstep.py manifest …` | train 13 bước / 3 video (1 bước bị kẹp `end`), val 7 bước / 2 video |
| local | `convert_htstep.py manifest` trên chính `download_log.jsonl` đó | **cùng sha256** (`ea159fd8…`, `65565dc7…`) |
| Colab CPU | Loader train (`LazySupervisedDataset._get_item`, processor WAVE-7B, không trọng số) trên 2 manifest | Mặc định: 5 bước của video AV1 lỗi `Video too large`. Với `MAX_VIDEO_MB=4096`: **13/13 train, 7/7 val** ra `pixel_values_videos (960, 1176)`, grid `[4, 12, 20]`, audio `(1, 128, 30000)`; 2.31 s/item (train, có video lossless), 1.26 s/item (val) |

Chưa chạy được: mọi thứ cần GPU (train, `VramCallback`, eval).

### Cách chạy notebook (Colab, GPU L4 hoặc A100)

1. Push: `git -C "D:\Học\KL\Code\Omni\Omni-fix" push origin htstep_lora`.
2. Mở `notebooks/omni_htstep_train.ipynb`, chọn runtime L4 hoặc A100.
3. Chạy lần lượt các ô:

| Ô | Việc |
| --- | --- |
| 1 | Mount Drive, clone `htstep_lora` (in commit). Cài `training/requirements.txt` + `decord==0.6.0 av==19.0.2 transformers==4.51.3`. In GPU / VRAM tổng bằng `nvidia-smi`; compute < 8.0 (T4) thì dừng |
| 2 | Tải WAVE-7B + BEATs (`nguyenminh04/omni-model`) và adapter (`nguyenminh04/omni-adapters`), `CHECK adapter_is_author` (sha256 so với `YunzeLiu/OmniRetriever-7B`), `patch_wave_rope.py` |
| 3 | Biến môi trường như YC2, chi tiết ở dưới bảng |
| 4 | `download_htstep.py`, rồi `convert_htstep.py manifest`. In số video / bước / số bị loại theo lý do, chép manifest + log lên Drive |
| 5 | Eval bản pt (0-shot) trên val nhỏ bằng `eval_youcookii.py --name HT-Step` |
| 6 | `bash training/train.sh --adam_beta2 0.95 …`, chi tiết ở dưới bảng |
| 7 | Eval checkpoint mới nhất; in R@1/5/10, NDCG@10 so với 0-shot |
| 8 | Tóm tắt (VRAM sau khi load, VRAM đỉnh khi train ở `BATCH_SIZE` đã đặt, giây / bước, VRAM đỉnh khi eval, lỗi), ghi `summary_<GPU>_bs<B>.json` lên Drive |

**Ô 3, biến môi trường:**
- Như YC2: `WAVE_PATH`, `BEATS_PATH`, `DATA_PATH`, `VIDEO_ROOT`, `LORA_CKPT` = adapter gốc, `OUTPUT_DIR` trên Drive, `LORA_ONLY=True`, `LORA_INIT_ONLY=True`, `USE_TUPLE_INFONCE=False`, `GRADIENT_CHECKPOINTING=True`, `BATCH_SIZE=8`, `GRAD_ACCUM=1`, `EPOCHS=3`, `LR=1e-4`, `NUM_GPUS=1`.
- Thêm: `N_TRAIN_VIDEOS`, `N_VAL_VIDEOS`, `MAX_STEPS` (mặc định 20 / 5 / 20), `TRAIN_SEED=42`, `MAX_VIDEO_MB=4096`.
- `RUN_NAME` đặt thư mục trên Drive.

**Ô 6, train:**
- `MAX_STEPS > 0` thì thêm `--max_steps N --save_strategy steps --save_steps N`; không thì `--save_strategy epoch`.
- Log được tee lên Drive.
- Kiểm khối "Trainable parameters by group" đúng một dòng `LoRA 6,881,280`.
- Đếm `[OOM] Skipping` và `[ERROR] Sample`, rồi in `vram.json`.

Đổi `BATCH_SIZE` ở ô 3 rồi chạy lại ô 6–8 để xem VRAM ở batch khác. Nên đổi cả `RUN_NAME`, hoặc xoá `OUTPUT_DIR`.

Gửi lại cho agent: `summary_*.json`, cùng `train.log` / `eval/*.log` nếu có lỗi.

### Ghi chú

- **Bản lossless rất nặng:** khoảng 1.5 MB mỗi giây video (1.42 GB cho 975 s), so với khoảng 0.05 MB/s của bản AV1 gốc. Vì thế mp4 để ở `/content/htstep` (đĩa local), không để trên Drive. Toàn bộ 386 video AV1 khi chuyển mã sẽ chiếm nhiều đĩa: chưa đo, cần xem trước khi chạy hết dữ liệu.
- **Item AV1 chậm hơn:** item của video chuyển mã nạp chậm hơn khoảng 2× (2.3 s so với 1 s trên CPU Colab). Loader có hạn 30 s mỗi mẫu.
- **DeepSpeed:** nếu đã cài thì `train.sh` dùng DeepSpeed ZeRO-0 như YC2. Nếu DeepSpeed lỗi trên Python của Colab, đặt `USE_DEEPSPEED=False` ở ô 3 để chạy Python thường.
- **Lossless chỉ kiểm theo số khung:** chưa so pixel giữa AV1 gốc và bản H.264 (decord trên Linux không đọc được AV1). Chỉ kiểm số khung / fps / độ dài.
