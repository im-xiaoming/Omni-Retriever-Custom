# Hướng dẫn chạy: LoRA Omni trên HT-Step bằng Colab Pro

Theo `OMNI_HTSTEP_LORA_COLAB.md`. Có hai notebook:

- `notebooks/omni_htstep_precut.ipynb`: cắt clip, chạy trên runtime **CPU**.
- `notebooks/omni_htstep_train.ipynb`: train và chấm, chạy trên runtime **A100**.

Nhánh: `htstep_lora`. Yêu cầu ghi `htstep-lora`, nhưng nhánh trên GitHub tên `htstep_lora`, nên cả hai notebook clone `htstep_lora`.

## Chuẩn bị (một lần)

### 1. Push nhánh

`git -C "D:\Học\KL\Code\Omni\Omni-fix" push origin htstep_lora`

### 2. Repo HF dataset private trong tài khoản của bạn

Đây là repo HF **duy nhất** pipeline ghi vào. Các repo `nguyenminh04/*` và `YunzeLiu/OmniRetriever-7B` chỉ được đọc.

1. Trên huggingface.co, đăng nhập đúng tài khoản sẽ giữ clip (ở đây là `GraceFeel`), rồi bấm **New → Dataset**.
2. Owner: chính tài khoản đó. Đặt tên (ví dụ `htstep-omni-clips`), chọn **Private**, rồi bấm **Create dataset**.
3. Nếu repo đã tạo ở chế độ Public: vào trang repo → **Settings** → **Change repository visibility** → **Make private**.
4. Không chạy đoạn `upload_folder(folder_path=".")` mà HF gợi ý trên trang repo mới: đoạn đó đẩy cả thư mục hiện tại lên repo. Pipeline tự upload đúng các file cần (`shards/`, `state/`, `meta/`).

Repo cần khoảng **45 GB**: shard clip khoảng 44 GB, cộng `meta/` và `state/`. Kiểm hạn mức lưu trữ private của tài khoản trong Settings → Billing / Storage.

Ô 1 của notebook precut chạy `precut_htstep.py check-repo`, và notebook **dừng** nếu:

- repo chưa có;
- repo không private;
- repo không nằm trong tài khoản của `HF_TOKEN`;
- token chỉ có quyền Read.

### 3. Token Write

1. huggingface.co → ảnh đại diện → **Settings** → **Access Tokens** → **Create new token**.
2. Token type: **Write**. Đặt tên, ví dụ `colab-htstep`. Bấm **Create token** rồi chép chuỗi `hf_…`.
3. Token Write của chính tài khoản đó đọc được các repo công khai như `nguyenminh04/*`, nên không cần token thứ hai.

Nếu chọn token **Fine-grained** thay vì Write:

- cấp quyền ghi cho repo `GraceFeel/htstep-omni-clips`;
- cấp quyền đọc các repo public / gated.

### 4. Colab Secrets

1. Mở notebook trên Colab, bấm biểu tượng chìa khoá (**Secrets**) ở thanh bên trái → **Add new secret**.
2. Name: `HF_TOKEN`. Value: chuỗi `hf_…` vừa tạo.
3. Bật **Notebook access** cho từng notebook (precut và train đều cần).

### 5. Ô 0 của hai notebook

- `HF_CLIPS` đã điền sẵn `GraceFeel/htstep-omni-clips`, và phải trùng nhau ở cả hai notebook. Đổi tên repo thì sửa ở cả hai.
- Notebook train có thêm `RUN` (tên thư mục lượt train), mặc định `lora_bs32`.

### 6. Drive và đĩa

| Chỗ | Cần |
| --- | --- |
| Drive trống | Khoảng **5 GB**: checkpoint (chỉ LoRA + optimizer, giữ 1 bản), embedding bench của pt và best, `omni_step_emb_*.npz`, log |
| Benchmark trên Drive | Phải có sẵn ở `MyDrive/Colab Notebooks/code KL/Omni_benchmark_data/` (`benchmark.zip`, `omniretriever_bench_final.jsonl`) |
| Đĩa Colab, runtime CPU | Mỗi worker giữ một mp4 tạm (bản H.264 lossless của video AV1 dài có thể tới khoảng 1.5 GB), cộng một shard 1.5 GB. Script dừng khi đĩa trống dưới 10 GB |
| Đĩa Colab, runtime A100 | Clip khoảng 44 GB, model khoảng 20 GB, benchmark. Ô 3 kiểm đĩa trước khi tải |

## Bước 1: cắt clip (`omni_htstep_precut.ipynb`, runtime CPU, High-RAM nếu có)

| Ô | Việc | Khi nào chạy |
| --- | --- | --- |
| 0, 1 | Cấu hình. Mount Drive, clone nhánh, cài `requirements-htstep.txt`, kiểm repo clip (`check-repo`). In bản thư viện, ffmpeg, nproc, RAM, đĩa | Mỗi phiên |
| 2 | Sinh split từ nhãn gốc `facebookresearch/htstep@cca1dea`. Kiểm `items_*.jsonl` trùng **từng byte** với bản đã commit (`CHECK items_unchanged`). In số video / bước theo lý do | Một lần |
| 3 | Kiểm chuyển mã trên 5 video AV1 ngắn nhất: khung decord đọc từ bản H.264 phải trùng từng pixel với khung ffmpeg giải thẳng từ AV1. Không đạt thì notebook dừng | Một lần |
| 4 | Cắt clip. Tự resume theo các manifest đã upload lên HF | Chạy lại mỗi khi phiên bị ngắt: mở lại notebook, chạy ô 0, 1 rồi ô 4 |
| 5 | Kiểm toàn vẹn, rồi sinh manifest train / val; upload `meta/` lên repo clip | Sau khi ô 4 báo `0 to cut` |

Ghi chú về ô 4:

- Mỗi video xong in một dòng: số clip, AV1 có chuyển mã không, kích thước shard đang gom, thời gian ước lượng còn lại.
- Đủ khoảng 1.5 GB clip thì gom thành `shards/clips_NNNN.tar` và upload. Thứ tự upload: tar trước, rồi manifest, rồi `state/done.jsonl` / `state/failed.jsonl`.
- **Phiên bị ngắt:** chỉ mất các video của shard đang gom dở (chưa upload); phiên sau cắt lại các video đó. Bấm Stop thì script upload phần đã xong rồi mới dừng.
- **`fetch_error`** (lỗi mạng sau 5 lần thử) được ghi vào `state/failed.jsonl`; lần chạy ô 4 sau sẽ thử tải lại.

Ghi chú về ô 5:

- Ô 5 dừng nếu còn bước chưa cắt (`missing`), có `id` nằm ở hai shard, hoặc thiếu tar của một manifest.
- In bảng số item theo lý do: `ok`, `past_end`, `unreadable`, `empty_video_track`, `transcode_mismatch`, `fetch_error`.

Gửi lại cho agent: thư mục Drive `MyDrive/omniretriever/htstep_lora/precut/`, gồm `split.log`, `check_transcode.json`, `verify.log`, `finalize.log`, `meta/` và log của ô 4.

## Bước 2: train (`omni_htstep_train.ipynb`, runtime A100)

| Ô | Việc | Khi nào chạy |
| --- | --- | --- |
| 0, 1 | Cấu hình, mount, clone, cài thư viện. Không phải A100 thì dừng | Mỗi phiên |
| 2 | Tải WAVE-7B + BEATs và adapter. Kiểm sha256 adapter với `YunzeLiu/OmniRetriever-7B`, khác thì dùng bản của YunzeLiu. Vá RoPE | Mỗi phiên |
| 3 | Tải toàn bộ shard + `meta/` từ repo clip, giải nén vào `/content/htstep_clips`, kiểm sha256 từng clip. Giải nén benchmark, tạo tập theo dõi 1 000 record, clone taxonomy | Mỗi phiên (khoảng 44 GB) |
| 4 | Đo VRAM với K = 1, 2, 4, 8, rồi chọn K. Ghi `chunk.json` + `protocol.json` lên Drive | **Một lần**; các lần sau tự bỏ qua |
| 5 | Mốc 0-shot của pt: `val_seen`, bench 12 hướng, `omni_step_emb_pt.npz` | Một lần; đã có kết quả thì tự bỏ qua |
| 6 | Train. Ô này tự resume | Chạy lại mỗi phiên cho đến khi in `train xong` |
| 7 | Chấm checkpoint `best` trên `val_seen` và bench, export `omni_step_emb_best.npz` | Sau khi train xong |
| 8 | Bảng pt / best, `summary.json` | Cuối cùng |

Ghi chú về ô 4:

- Chọn K lớn nhất có `max_memory_reserved` dưới 90 % VRAM.
- Đặt eval mỗi 1/2 epoch nếu một lượt val tốn hơn 15 % thời gian một epoch; không thì mỗi 1/4 epoch.
- In VRAM sau khi load, VRAM theo từng K, VRAM khi eval, thời gian một epoch và số phiên Colab cần.

Ghi chú về ô 6:

- Ô 6 kiểm `protocol.json` trước khi train, rồi gọi `training/train.sh` với batch 32, `--grad_cache_chunk K`, 2 epoch, lr 1e-4, seed 42.
- Lưu checkpoint mỗi 200 bước và sau mỗi lần eval. Lượt đầu nạp adapter pt; lần sau resume từ checkpoint cuối.
- In `CHECK lora_only` (phải có đúng một dòng `LoRA 6,881,280`), VRAM và lịch sử eval.

Kết quả nằm ở `MyDrive/omniretriever/htstep_lora/<RUN>/`:

- `ckpt/` gồm `checkpoint-N` (last), `best/` (chỉ LoRA), `metrics.json`, `eval_state.json`, `vram.json`;
- `chunk.json`, `vram_probe.json`, `protocol.json`;
- `eval/metrics_htstep_{pt,best}.json`, `eval/bench/bench_{pt,best}.json`;
- `omni_step_emb_{pt,best}.npz`, `summary.json`.

Gửi lại cho agent: `summary.json`, `chunk.json`, `vram_probe.json`, `ckpt/metrics.json`, và các file `*.log` nếu có lỗi.
