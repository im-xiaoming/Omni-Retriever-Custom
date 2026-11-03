# Nhật ký: LoRA Omni trên HT-Step, chạy trên Colab Pro

Theo `OMNI_HTSTEP_LORA_COLAB.md` (2026-11-03). Hướng dẫn chạy: `docs/htstep/HUONG_DAN_COLAB.md`.

## 2026-11-03: agent (chỉ local; phần chạy trên Colab chờ user)

**Nhánh:** `htstep_lora`. Yêu cầu ghi `htstep-lora`, còn nhánh trên GitHub tên `htstep_lora`, nên notebook clone `htstep_lora`. Chưa push.

### Commit

| Commit | Nội dung |
| --- | --- |
| `f81a242` | Dọn code thử nghiệm HT-Step (mục 1) |
| `91cef56` | Dữ liệu: `precut_htstep.py` viết lại; `convert_htstep.py` `finalize` / `split`; `requirements-htstep.txt` |
| `c562766` | Train: checkpoint chỉ LoRA, lưu sau mỗi eval, `probe_htstep.py` chọn K, `eval_htstep.py`, `make_protocol.py` |
| `c7d5ddf` | Hai notebook Colab Pro |
| (commit này) | `HUONG_DAN_COLAB.md`, nhật ký này, dòng đầu `TAI_VIDEO.md` |

### Mục 1: file xoá / giữ

So với `chien` (`origin/chien` = `460257f`). `grep` toàn repo trước khi xoá: không còn chỗ nào gọi tới các file bị xoá. Hai chỗ còn import `range_get` (`convert_htstep.py` lệnh `smoke`, `download_htstep.py`) cũng bị xoá cùng.

| File | Quyết định | Lý do |
| --- | --- | --- |
| `notebooks/omni_htstep_smoke.ipynb`, `scripts/smoke_eval.py`, `data/htstep/smoke/*`, `docs/htstep/NHAT_KY_SMOKE.md` | xoá | Smoke test |
| `notebooks/omni_htstep_lora.ipynb` | xoá | Notebook pilot / Vast, thay bằng hai notebook mới |
| `notebooks/omni_htstep_train.ipynb` (bản cũ, tải mp4 kiểu YC2) | thay | Cùng tên với notebook train mới |
| `scripts/download_htstep.py`, `docs/htstep/NHAT_KY_TRAIN_COLAB.md`, lệnh `smoke` / `manifest` của `convert_htstep.py` | xoá | Hướng tải mp4 kiểu YC2, đã bỏ. `probe` / `same_video` / lệnh chuyển mã chuyển sang `htstep_io.py` |
| `scripts/run_vast.sh`, `docs/htstep/VAST.md` | xoá | Vast. Không còn cờ `PRECUT_PILOT` (chỉ có trong notebook đã xoá) |
| `scripts/bench_precut_one_tar.py`, `scripts/probe_codecs_htstep.py`, `scripts/inventory_htstep_videos.py` | xoá | Script đo một lần. `range_get` chuyển sang `src/omniretriever/data/htstep_io.py` |
| `scripts/eval_youcookii.py` (thêm `--name`, VRAM), ngưỡng `MAX_VIDEO_MB` trong `data_qwen.py` | trả về như `460257f` | Chỉ phục vụ hướng YC2 đã bỏ. Mục 1.4: không đổi hành vi code YC2 |
| `docs/htstep/TAI_VIDEO.md`, `docs/htstep/tai_video/*` | giữ | Báo cáo và số liệu. Thêm dòng đầu: script đo đã xoá ở `f81a242` |
| `docs/htstep/UNIAV_XU_LY_DU_LIEU.md` | giữ, **không** thêm dòng đầu | Script của báo cáo này nằm ngoài repo (`D:\Học\KL\Data\Htstep\khao_sat\`) và không bị xoá, nên câu "đã xoá" sẽ sai |
| `data/htstep/{tar_index.csv, items_*.jsonl, htstep_split_omni.json}`, `docs/htstep/tai_video/codecs.jsonl` | giữ | Pipeline đọc. `codecs.jsonl` dùng để chọn video AV1 cho lệnh `check-transcode` |
| `scripts/{htstep_common, export_step_emb, eval_bench, make_protocol, materialize_adapter}.py`, `training/qwenvl/train/{grad_cache, retrieval_eval}.py`, `src/omniretriever/data/clip.py`, `VramCallback`, nhánh `clip` của `data_qwen.py`, `USE_DEEPSPEED` của `train.sh` | giữ | Pipeline cuối dùng |
| `scripts/probe_htstep.py` | giữ, viết lại | Thành ô 4: chọn K |

### Thay đổi so với code cũ

| File | Chỗ | Việc |
| --- | --- | --- |
| `src/omniretriever/data/htstep_io.py` | mới | `range_get`, `hf_get`, `probe`, `same_video`, `transcode`, `file_sha256`, `upload`. Lỗi mạng retry `5·2**k`, tối đa 5 lần |
| `scripts/precut_htstep.py` | viết lại (503 dòng) | Chi tiết ở dưới bảng |
| `scripts/convert_htstep.py` | `:220` | `--yc2` nhận file hoặc danh sách id (Colab không có file YC2) |
| `scripts/convert_htstep.py` | `:256`, `:267`, `:284` | Record có thêm `src_transcoded` |
| `scripts/convert_htstep.py` | `:290-295` | `dropped_steps.json` có thêm `videos` / `videos_by_reason`: video bị loại cả video lúc cắt |
| `training/qwenvl/train/trainer.py` | `:204-210` | Khi `--grad_cache_chunk > 0` và chỉ tensor `lora_` được train, `adapter_model.safetensors` của checkpoint chỉ giữ LoRA (head là bản pt). Lúc resume, `Trainer._load_from_checkpoint` → `load_adapter` nạp LoRA đè lên adapter pt. YC2 (`grad_cache_chunk = 0`) không đổi |
| `training/qwenvl/train/retrieval_eval.py` | `:211` | `should_save` sau mỗi lần eval |
| `scripts/probe_htstep.py` | viết lại | Chi tiết ở dưới bảng |
| `scripts/eval_htstep.py` | mới | Chấm `val_seen` bằng đúng `encode` / `score` của eval lúc train. In R@1/5/10, NDCG@10, MRR, R@1 multi. Ghi `metrics_htstep_<tag>.json`, `val_embeds_<tag>.npz` |
| `scripts/make_protocol.py` | `--data-dir` / `--out` | Giá trị mục 2 của yêu cầu mới. `check` báo nếu commit code khác commit lúc ghi protocol (không coi là FAIL) |
| `requirements-htstep.txt` | mới | Ghim `decord==0.6.0`, `av==19.0.2`, `transformers==4.51.3`; thêm `peft==0.15.2`, `accelerate==1.6.0` cùng đợt phát hành; `deepspeed`, vì `trainer.py:40` import nó, dù lượt HT-Step chạy với `USE_DEEPSPEED=False` |
| `tests/test_htstep_lora.py` | | Bỏ test smoke / manifest / download. Thêm test `same_video`, precut (`read_items`, tên shard, thứ tự upload, `index_rows`, `cut_video` + `finalize`), `plan` của probe, callback eval (lưu / dừng sớm / resume) |

Phần có sẵn từ trước, giữ nguyên:

- `data_qwen.py`:
  - `:501` `process_clip`, `:758`, `:808` nhánh `clip`;
  - `:939`, `:1292` mang `text` / `id` tới loss.
- `trainer.py:1068`, `:1143`: GradCache.
- `argument.py:94`: các cờ.
- `train_qwen.py`:
  - `:107` `VramCallback`;
  - `:254` `TRAIN_SEED`;
  - `:292` `InstanceCollator`;
  - `:513` eval callback;
  - `:705` `ATTN_IMPL`.
- `train.sh:130-132`: `USE_DEEPSPEED`.

**`precut_htstep.py`:**

- **`run`:**
  - Đơn vị là video. Train: range read member trong tar; val: file của `htstep-val-seen-videos`.
  - ffprobe; AV1 thì chuyển mã bằng đúng lệnh ở mục 3, rồi kiểm số khung, `avg_fps` và độ dài.
  - Video không đọc được bị loại với lý do `empty_video_track`, `transcode_mismatch`, hoặc `unreadable` (mọi bước lỗi).
  - Cắt từng bước bằng `cut_clip`; lý do theo bước là `past_end`, `unreadable`. mp4 tạm xoá trong `finally`.
  - Đủ `--shard-gb 1.5` thì gom `shards/clips_NNNN.tar` + `.manifest.jsonl` (id, status, sha256, bytes, window, `video_length`, `src_transcoded`). Upload tar trước, manifest sau, rồi `state/done.jsonl`, `state/failed.jsonl`; bản local chỉ xoá sau khi upload xong.
  - Resume lấy manifest trên HF làm trạng thái thật, nên một video chỉ nằm ở đúng một shard. Tên shard không bao giờ dùng lại, nên không ghi đè shard nào.
  - Kiểm đĩa trống trước khi giao mỗi video. Số worker mặc định = min(nproc, RAM GB / 2.5). Bấm Stop thì upload phần đã xong rồi mới dừng.
- **`verify`:**
  - Mỗi `id` đúng một dòng; không `id` nào ở hai shard; mỗi manifest có tar trên HF.
  - In bảng theo split / lý do. Ghi `precut_index.jsonl` và `_summary.json`.
- **`check-transcode`:**
  - Lấy 5 video AV1 ngắn nhất, 8 khung trải đều mỗi video.
  - So khung decord (bản H.264) với khung ffmpeg giải thẳng từ AV1. Bên ffmpeg dùng `scale=W:H:flags=bilinear,format=rgb24`, bộ lọc decord dựng (`decord src/video/ffmpeg/filter_graph.cc`, `threaded_decoder.cc`; mặc định của vf_scale ở FFmpeg 4 là bilinear).
  - Ghi thêm `yuv_equal` (YUV của AV1 so với H.264) để chẩn đoán: nếu RGB lệch mà YUV trùng thì là do chuyển màu, không phải do mã hoá.
- **`fetch`:**
  - Kiểm đĩa trống, cần tổng shard + một tar + 5 GB.
  - Tải và giải nén từng shard, kiểm sha256 từng clip, tải `meta/`. Chạy lại thì bỏ shard đã lấy (`fetched.json`).

**`probe_htstep.py`:**

- K = 1, 2, 4, 8; mỗi K 3 bước optimizer, batch 32, GradCache, gradient checkpointing, bf16. Bước đầu là khởi động, không tính giờ.
- Inference batch 1 trên 50 item val.
- Chọn K lớn nhất có `max_memory_reserved` dưới 90 % VRAM.
- Thời gian một bước lấy max(GPU, thời gian nạp 32 item / 4 worker).
- Luật eval: 1/2 epoch nếu một lượt val dài hơn 15 % một epoch, không thì 1/4.
- Ước lượng giờ cả lượt và số phiên Colab, giả định 12 giờ một phiên (`--session-hours`).
- Ghi `chunk.json`, `vram_probe.json`.

### Chỗ làm khác hoặc chưa rõ trong yêu cầu

1. **Không có shard `.partial`.**
   - Manifest đã upload là trạng thái thật; shard đang gom dở khi phiên chết sẽ được cắt lại (tối đa khoảng 1.5 GB clip).
   - Bấm Stop thì shard dở vẫn được upload như một shard bình thường, nhỏ hơn.
   - Mục 3 không bắt buộc `.partial`; cách này tránh trường hợp một `id` nằm ở hai shard.
2. **Split trên Colab.**
   - Ô 2 dùng `annotations.json` gốc của GitHub (`cca1dea`), sha256 `e40d8a2b…`. File local đã bị ai đó thêm trường `headline` (sha256 `ba536050…`).
   - Luật chọn không đọc trường `headline`, nên `items_*.jsonl` phải trùng **từng byte**; ô 2 kiểm điều này.
   - `htstep_split_omni.json` sinh trên Colab sẽ khác bản commit ở sha256 của `annotations.json`, ở `tar_index_check` (`--no-verify`, vì file kích thước tar chỉ có ở local) và ở mục UniAV (file chỉ mục đặc trưng chỉ có ở local).
   - Bản đẩy lên `meta/` và băm vào protocol là bản sinh trên Colab.
3. **Video bị loại lúc cắt** ghi ở `dropped_steps.json` (`videos`), không ghi lại vào `excluded` của split. Lý do: split đã cố định trước khi cắt, và precut đọc nó.
4. **Mốc pt trên `val_seen` chấm hai lần:** ô 5 (`eval_htstep.py`) và bước 0 của callback lúc train (cần để có mốc `best`). Hai số phải trùng; ô 8 in cả hai để kiểm.
5. **Không còn kiểm tương đương embedding clip ↔ mp4 trên GPU** (plan cũ mục 3.4). Yêu cầu mới không có bước này, và máy A100 không có mp4. Tương đương ở mức input do `test_clip_matches_mp4_path` / `test_process_clip_in_data_qwen` kiểm (CPU, mp4 tự sinh).
6. **Không thấy best:** nếu không lần eval nào vượt pt thì không có `best/`. Ô 7 ghi lỗi đó và bỏ qua bench / export cho best.
7. **Bench:** `eval_bench.py` in N thực tế. Ô 8 ghi N; nếu N khác 3 782 thì không so thẳng với Bảng 4 của bài báo.

### sha256

| File | sha256 |
| --- | --- |
| `data/htstep/htstep_split_omni.json` (bản commit, sinh từ `annotations.json` local) | `e354a9c02dfe1bc7eb31fdb19eebeccb18cea7bf0bf1b9e2f67de47c8814a3d3` |
| `data/htstep/items_train.jsonl` (44 151 bước / 7 110 video) | `4ad730345a2ccd10faae3f4f9c4a34f67179448b28fff608f436b9a4a7b0cb22` |
| `data/htstep/items_val.jsonl` (2 990 bước / 510 video) | `e51fdc8f69feeddfce7d3b51f3f3063ce9272f09f67b39b86e359bb20068e6dd` |
| `data/htstep/tar_index.csv` | `6865b0772402ef2aba945d8b2b919383272d8d5bfd317eea38fd641c03ff7853` |
| `requirements-htstep.txt` | `70d3fb7982e2ad06df1d014a771dc6c1da3f0bee0db1c02ee0de20b4c1ce9cf2` |
| `protocol.json` | chưa có: ghi ở ô 4 của notebook train, trên Drive |
| `train_omni_htstep.jsonl` / `val_omni_htstep.jsonl` | chưa có: sinh ở ô 5 của notebook precut |

### Test

`pytest tests/test_htstep_lora.py -q` chạy ở local: **17 passed**.

- Môi trường: conda env `omni`, gồm Python 3.10.21, torch 2.14 CPU, decord 0.6.0, av 17.1.0 (không phải bản ghim 19.0.2), librosa 0.11; cộng transformers 4.51.3 / huggingface_hub 0.30.2 nạp qua `PYTHONPATH` từ một thư mục `--target` trong scratchpad.
- Không có ffprobe ở local, nên test `cut_video` thay `probe` bằng giá trị giả. Đường ffprobe / ffmpeg thật chỉ chạy ở ô 3 / ô 4 của notebook precut.

| Nhóm | Test |
| --- | --- |
| Luật chọn, text / id, split tái lập | `test_selection_rules` (hơn 48 bước, `end <= start`), `test_text_and_key`, `test_split_reproducible`, `test_items_against_independent_count` (dữ liệu thật) |
| Mask, loss | `test_mask_and_loss_by_hand` |
| GradCache | `test_grad_cache_matches_full_batch` (chunk 1 / 3 / 8, encoder nhỏ có dropout, sai lệch tương đối < 1e-5) |
| Loader `clip` | `test_clip_matches_mp4_path`, `test_clip_window_rules` (kẹp `end`, `past_end`), `test_process_clip_in_data_qwen` |
| Precut / resume / shard | `test_transcode_check`, `test_precut_read_items_and_names` (không dùng lại tên shard), `test_precut_shard_flush` (tar trước manifest, xoá local sau upload), `test_precut_index_rows` (trùng shard, thiếu, `fetch_error`), `test_cut_video_and_finalize` |
| Probe, callback | `test_probe_plan`, `test_eval_callback_save_stop_resume` |
| YC2 không đổi | `test_convert_youcookii_untouched`: `convert_youcookii.py` và `eval_youcookii.py` trùng `460257f`, nên manifest YC2 sinh ra giống byte-by-byte |

### Chưa chạy (cần Colab)

Toàn bộ hai notebook. Gửi lại: thư mục `precut/` trên Drive (bước 1) và `summary.json`, `chunk.json`, `vram_probe.json`, `ckpt/metrics.json` (bước 2), theo `HUONG_DAN_COLAB.md`.

## 2026-11-03 (2): yêu cầu cập nhật, repo clip của user

Yêu cầu sửa mục 3 / 7 / 8:

- Repo clip là repo dataset **private trong tài khoản của user**, do user tự tạo, dùng token Write của chính tài khoản đó.
- Không ghi vào repo HF nào khác; `nguyenminh04/*` chỉ đọc.
- Hướng dẫn phải có cách tạo repo, tạo token và đặt Secrets.

User đã tạo `GraceFeel/htstep-omni-clips`. Kiểm qua API công khai lúc 2026-11-03: repo có `"private": false`, nội dung chỉ có `.gitattributes`. Cần chuyển sang private trước khi chạy; ô 1 sẽ dừng nếu chưa.

| File | Việc |
| --- | --- |
| `scripts/precut_htstep.py` `check_clip_repo`, lệnh `check-repo` | Bỏ `create_repo`, nên pipeline không bao giờ tạo repo. Trước khi ghi, kiểm: repo đã có, private, owner = tài khoản của `HF_TOKEN`, token không phải Read. `run` gọi kiểm này đầu tiên |
| `notebooks/omni_htstep_precut.ipynb` ô 0, 1 | `HF_CLIPS = "GraceFeel/htstep-omni-clips"` (ở ô cấu hình, không có trong script). Ô 1 chạy `check-repo` và dừng nếu sai |
| `notebooks/omni_htstep_train.ipynb` ô 0, 6 | Cùng `HF_CLIPS`. Bỏ tuỳ chọn `HF_RUN_REPO` / `--hf_run_repo`, vì nó sẽ tạo và ghi một repo HF thứ hai. Code `hf_run_repo` của `RetrievalEvalCallback` vẫn còn nhưng mặc định rỗng và notebook không dùng |
| `docs/htstep/HUONG_DAN_COLAB.md` | Mục "Chuẩn bị": tạo repo private (hoặc chuyển public → private), không chạy đoạn `upload_folder(".")` HF gợi ý, tạo token Write (hoặc Fine-grained), Colab Secrets, Drive / đĩa |
| `tests/test_htstep_lora.py` `test_clip_repo_check` | Repo public / token Read / repo chưa có / repo tài khoản khác đều dừng; private + Write hoặc Fine-grained thì qua |

Test local: **18 passed** (cùng môi trường như trên).
