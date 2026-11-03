# Tải mp4 HT-Step và cắt clip: kiểm kê, đo, ước lượng

> Các script đo trong báo cáo này (`inventory_htstep_videos.py`, `bench_precut_one_tar.py`, `probe_codecs_htstep.py`) đã xoá ở commit `f81a242`; số liệu vẫn đúng. `range_get` chuyển sang `src/omniretriever/data/htstep_io.py`.

Ngày: 2026-11-02. Phạm vi: chỉ kiểm tra và đo, **không** cắt toàn bộ dữ liệu, không sửa plan
(`OMNI_HTSTEP_LORA.md` mục 2.2 / 3). Nhãn:
- **[đo thực tế]**: số lấy từ script trong phụ lục. Output nằm ở `docs/htstep/tai_video/`.
- **[ước lượng]**: suy ra từ số đo, có ghi cách tính.
- **[đọc code]**: dẫn `file:dòng`.
- **[chưa xác minh]**: chưa kiểm được, có ghi lý do.

## Tóm tắt (10 dòng)

1. Trên HF có tổng cộng **187.1 GB**: 113 tar (179.6 GB) và 330 mp4 val (7.57 GB). Cần tải **183.7 GB**, tức 98 %; bỏ được rất ít [đo thực tế].
2. 7 453 video trong tar = 7 110 train + 181 val + 162 bản val trùng (dùng bản ở val repo). Không tar nào chứa video thuộc subset khác [đo thực tế].
3. Range-read (b) chỉ bớt được **1.7 GB** byte tải so với tải cả tar (a). Lợi thật của (b) là đĩa: lúc cao nhất **0.47 GB so với 2.45 GB** cho một tar [đo thực tế].
4. Hai cách cho clip **giống hệt nhau** (558/558 bước, cùng sha mảng), cả trên Colab lẫn local [đo thực tế].
5. Cắt tốn **1.20 giây CPU mỗi bước** trên Colab free (2 vCPU). Một tar (558 bước, 1.97 GB) mất 426 s (b) / 442 s (a). Mạng không phải nút thắt (112 MB/s) [đo thực tế].
6. Toàn bộ 47 141 bước: khoảng **10–11 giờ** trên một phiên Colab free, khoảng **16 giờ vCPU**; clip ra khoảng **41 GB** (0.944 MB/bước × 43 676 bước đọc được) [ước lượng].
7. **Vấn đề lớn nhất:** decord 0.6.0 (pip, Linux) **không đọc được video AV1**. Tar 4007 có 6/82 video AV1 và 2 video h264 không giải mã được, mất 60/558 bước (10.8 %). Toàn bộ dữ liệu (ffprobe): mất **464/7 110 video train (7.6 % bước)** và **20/510 video val (3.9 % bước)**. Codec là [đo thực tế]; việc decord lỗi suy từ codec: đã thử 7 file AV1 + 2 file track rỗng, đều lỗi.
8. Cắt trên Windows (decord / PyAV khác bản) cho 38/498 clip **khác byte** so với Linux, và đọc được AV1. Vì vậy **phải cắt trên Linux**, với đúng bản thư viện của máy train [đo thực tế].
9. Đề xuất cách **(b)**: range-read từng mp4, resume theo video, đóng shard 1–2 GB upload ngay. Chạy trên Colab free (2 phiên chia đôi danh sách), hoặc một máy Vast khoảng 20 nhân / ≥ 50 GB RAM, khoảng 1–2 giờ, khoảng $1–3 [ước lượng].
10. Cần bạn quyết (mục 8): xử lý video AV1 / file hỏng thế nào; ghim bản `decord` / `av`; chấp nhận clip khoảng 41 GB.

---

## 1. Kiểm kê (chỉ HF API và header tar)

Script: `scripts/inventory_htstep_videos.py`. Output: `tai_video/inventory.json`, `tai_video/inventory_per_tar.csv`, `tai_video/inventory.out.txt`.

### 1.1 Chỉ mục tar còn đúng không [đo thực tế]

| Kiểm | Kết quả |
| --- | --- |
| Revision `htstep-videos` hiện tại so với revision lúc lập `tar_index.csv` | cùng `541f392c` |
| Số tar trên HF / trong chỉ mục | 113 / 113 |
| Tar có size khác chỉ mục | 0 |
| Đọc lại header 512 byte của **member cuối** mỗi tar (range read): tên + size khớp chỉ mục | 113/113 khớp |
| `offset + size` ≤ size tar | đúng với mọi tar |
| Member trong chỉ mục | 7 453 (đều là mp4) |

Chỉ mục còn đúng, không cần đọc lại toàn bộ header. Bản sao đã commit tại `data/htstep/tar_index.csv`, sha256 `6865b077…7853`, trùng với `pseudocap/metadata/tar_index.csv`.

### 1.2 Dung lượng [đo thực tế]

| Nguồn | Số file | Byte | Ghi chú |
| --- | ---: | ---: | --- |
| `htstep-videos/videos/*.tar` | 113 | 179 550 023 680 (179.55 GB) | tar nhỏ nhất 18.9 MB, trung vị 1.96 GB, lớn nhất 2.58 GB |
| `htstep-val-seen-videos` (mp4) | 330 | 7 567 371 664 (7.57 GB) | 329 file dùng; bỏ 1 file `part1/J2nZX27tbbo.f605.mp4` (5.3 MB, bản phụ của yt-dlp) |
| **Tổng trên HF** | | **187.12 GB** | |

Bảng từng tar (113 dòng) nằm ở `tai_video/inventory_per_tar.csv`, gồm các cột `tar_bytes`, `members`, `need_bytes`, `need_frac`, và số video / byte theo từng lý do.

### 1.3 Mỗi member thuộc nhóm nào (luật 2.2, theo `data/htstep/htstep_split_omni.json`) [đo thực tế]

| Nhóm | Video | Byte |
| --- | ---: | ---: |
| train (subset `train`, có mp4, không trùng YC2) | 7 110 | 172.50 GB |
| val (`val_seen`, chỉ có bản trong tar) | 181 | 3.65 GB |
| bản val trùng: `val_seen` có cả trong `htstep-val-seen-videos`, nên **không** tải bản tar | 162 | 3.39 GB |
| subset khác / không có trong `annotations.json` / trùng YC2 / không còn bước | 0 | 0 |
| **Cần tải từ tar** | 7 291 | **176.14 GB** |
| **Cần tải từ val repo** | 329 | **7.56 GB** |
| **Tổng cần tải** | 7 620 | **183.70 GB** |

- `6MBctYaMU8U` (trùng YC2) không nằm trong tar nào, nên không tốn byte (`excluded: yc2_overlap` trong split).
- Có đúng 162 video `val_seen` có hai bản (`val_videos_with_two_copies = 162`). Split đã chọn bản val repo cho cả 162 (`source.repo = htstep-val-seen-videos`).
- Tar không cần byte nào: chỉ `htstep_videos_1001.tar` (86 member, đều là bản val trùng, 1.69 GB).
- Tar cần ít: `htstep_videos_1003.tar` (2/78 member cần, 3.5 % byte). 111 tar còn lại cần ≥ 99.9 % byte.
- Theo bước: 47 141 bước (train 44 151 + val 2 990), trung bình **3.9 MB mp4 mỗi bước**. Tar 4007 dùng để đo: 3.5 MB mỗi bước.

---

## 2. Code tải có sẵn: phần dùng lại được [đọc code]

### 2.1 Pipeline pseudo-caption (`D:\Học\KL\Data\Htstep\pseudocap`)

| Chức năng | Chỗ | Dùng lại? |
| --- | --- | --- |
| Đọc header tar qua HTTP range: resolve redirect một lần, **bỏ token khi gọi CDN**, retry backoff `2**k` | `tools/build_tar_index.py:26-45` | Có: `inventory_htstep_videos.py` import `range_reader` |
| Parse header (GNU long name, pax `path`) | `tools/build_tar_index.py:48-68` | Có, khi cần lập lại chỉ mục |
| Resume theo tar (tar cùng size thì không đọc lại); ghi file sau mỗi tar | `tools/build_tar_index.py:83-90`, `97-104` | Ý tưởng dùng lại |
| `range_get`: range read một member ra file, kiểm `206` và đủ byte, resolve URL CDN **mỗi video** (URL ký có hạn), retry `5·2**k` | `notebooks/htstep_pseudocap_AZ.ipynb` ô 6, dòng 88-108 | Có: chép gần nguyên vào `bench_precut_one_tar.py::range_get` |
| `fetch_videos(how='tar'\|'range')`. Mode tar: `hf_hub_download` (có resume `.incomplete`) rồi `extract` **từng member**. Mode range: từng video | ô 6, dòng 122-145 | Có |
| Chọn mode tự động: range khi tar > 3× byte cần, hoặc khi < 20 % member cần | ô 6, dòng 147-153 | Với HT-Step chỉ tar 1001 / 1003 rơi vào range (mục 1.3) |
| `need_disk`: dừng nếu thiếu đĩa (cần + biên) | ô 6, dòng 84-86 | Có |
| Trạng thái: `videos_done.txt`, `failed.csv`, `tars.json` (`claim_tar`: `in_progress` + `session_id` + `STALE_HOURS` để nhiều phiên không giẫm nhau) | ô 1, dòng 85-124; ô 6, dòng 110-120 | Có: thiết kế resume mục 5 dựa vào đây |
| `finally: rmtree(work)` để xoá tar và mp4 tạm kể cả khi lỗi | ô 6, dòng 204-206 | Có |

### 2.2 UniAV `origin/cleanup-htstep` @ `7c415ac` (đọc bằng `git show`, không checkout)

| Chức năng | Chỗ | Dùng lại? |
| --- | --- | --- |
| `download_hf.py`: tải tar, `extractall` vào `<chunk>.tmp`, ghi `.complete` rồi `rename`, nên một chunk trên đĩa luôn đủ; chạy lại thì bỏ chunk đã có | `tools/pipeline/download_hf.py:56-71`, `79` | Ý tưởng "ghi xong mới đổi tên" dùng cho shard |
| `max_ahead`: dừng tải khi đã có N chunk chờ trích (đĩa nhỏ) | `download_hf.py:33-35`, `51-59` | Không cần với (b): mỗi worker chỉ giữ một mp4 |
| `run.sh`: `SMALL_DISK=1` thì `--max-ahead 3 --delete-videos`; `JOBS` (đỉnh đĩa khoảng 2× tar mỗi job) | `tools/pipeline/run.sh:4-6`, `16-19` | Ghi chú đỉnh đĩa khớp số đo (a) |
| Chia phần theo máy: `shards_part1_{pc,colab,kaggle0,kaggle1}.csv`, `rest_{pc,colab,vast}.csv` (danh sách chunk riêng, không trùng) | `tools/pipeline/*.csv` | Cách chia "mỗi máy một file danh sách" dùng lại |
| `--upload-every N`: upload tar đặc trưng **một phần** sau mỗi N video, kèm `partial.txt` (id còn thiếu) | `extract_features.py:65`, `272-297` | Dùng lại cho shard đang đầy dở |
| `upload_partial.py`: **không ghi đè tar trên HF đã đầy hơn** (đếm từ README) | `tools/pipeline/upload_partial.py:11`, `17` | Dùng lại: so số item trong manifest shard trước khi upload |
| Marker `.uploaded/<chunk>` để chạy lại không upload lần nữa | `extract_features.py:296`, `317` | Có |

### 2.3 Omni-fix hiện có (`scripts/precut_htstep.py`, commit 78eade3)

- `fetch` (`precut_htstep.py:52-74`) tải cả tar rồi **giải nén mọi member cần trước khi cắt**. Đỉnh đĩa mỗi job vì thế khoảng tar + toàn bộ mp4 cần, tức khoảng 2× tar (≤ 5.2 GB) [ước lượng: 2 × 2.58 GB].
- Resume theo **tar** (`state/<unit>.json`, `index/<unit>.jsonl` trên HF) ở `:170-184`. Mỗi tar là một shard, khoảng 0.47 GB (`:125-130`), nhỏ hơn mức 1–2 GB yêu cầu. Upload có retry (`:77-90`).
- Hàm cắt `cut_clip` (`src/omniretriever/data/clip.py:51-76`) dùng nguyên cho cả hai cách đo bên dưới.

---

## 3. Đo trên một tar thật: (a) so với (b)

Script: `scripts/bench_precut_one_tar.py`.
- Tar: `videos/htstep_videos_4007.tar`, là tar trung vị trong các tar train: 82 video train, 558 bước, 1 971 261 216 byte cần (= 99.99 % tar).
- Cả hai cách cắt bằng `cut_clip` (8 khung `video_decord`, resize `min = max = 50176`, `video_length`, wav ≤ 8 s int16) rồi ghi `np.savez_compressed`, đúng như `precut_htstep.py`.
- (a): `hf_hub_download` cả tar, chép byte từng member ra mp4 tạm, cắt, xoá mp4; xoá tar ở cuối.
- (b): mỗi worker range-read đúng byte của một mp4, cắt, xoá.
- Một thread đo tổng byte trong thư mục làm việc mỗi 0.25 s (đỉnh đĩa).

### 3.1 Kết quả [đo thực tế]

| | Colab free, j=2, (a) | Colab free, j=2, (b) | Colab, j=4, (a) | Colab, j=4, (b) | Local, j=3, (a) | Local, j=3, (b) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Máy | 2 vCPU Xeon 2.2 GHz, 12.7 GB RAM, Linux | ″ | ″ | ″ | Windows 11, 8 nhân / 16 luồng, 15.4 GB RAM (3.6 GB trống) | ″ |
| Thời gian tải | tar 17.6 s (112 MB/s) | tổng 57.4 s trên các worker (khoảng 34 MB/s mỗi luồng) | tar 11.3 s | 71.2 s | tar 464 s (4.2 MB/s) | 897 s (khoảng 3.6 MB/s mỗi luồng) |
| **Wall cả tar** | **442 s** | **426 s** | 479 s | 460 s | 562 s | **435 s** |
| Wall pha cắt | 424 s | (đan xen với tải) | 467 s | | 98 s | (đan xen) |
| CPU cắt (tổng) | 670 s, 1.20 s/bước | 670 s | 702 s | 706 s | 352 s, 0.63 s/bước | 300 s, 0.54 s/bước |
| Cắt mỗi video p50 / p90 / max | 8.4 / 19.6 / 36.8 s | 8.0 / 19.4 / 36.6 s | 17.2 / 40.7 / 80.6 s | 16.0 / 41.2 / 78.1 s | 2.6 / 5.7 / 14.3 s | 2.1 / 4.8 / 10.1 s |
| Cắt mỗi bước mean / p50 / p90 / max | 1.42 / 1.16 / 2.98 / 9.0 s | 1.37 / 1.06 / 2.92 / 9.3 s | 2.96 / 2.41 / 6.19 / 16.8 s | 2.94 / 2.33 / 6.03 / 17.8 s | 0.45 / 0.37 / 0.85 / 2.0 s | 0.38 / 0.32 / 0.70 / 1.8 s |
| **Đĩa lúc cao nhất** | **2.45 GB** | **0.47 GB** | 2.46 GB | 0.49 GB | 2.49 GB | 0.53 GB |
| Clip ra | 470.3 MB (498 bước, 0.944 MB/bước) | 470.3 MB | 470.3 MB | 470.3 MB | 511.1 MB (542 bước) | 511.1 MB |
| RAM đỉnh mỗi worker | 2.18 GB | 2.20 GB | 2.10 GB | 2.29 GB | 1.21 GB | 1.17 GB |
| Bước ok / past_end / unreadable | 498 / 0 / 60 | 498 / 0 / 60 | 498 / 0 / 60 | 498 / 0 / 60 | 542 / 1 / 15 | 542 / 1 / 15 |
| Lỗi tải | 0 | 0 | 0 | 0 | 0 | 0 |
| Clip (a) = (b) (sha mảng) | 558/558 | | 558/558 | | 558/558 | |

Đọc bảng:
- **Colab bị giới hạn bởi CPU, local bị giới hạn bởi mạng.** Trên Colab, j=4 chậm hơn j=2 (2 vCPU), nên dùng **j = số vCPU**. Trên local, (b) nhanh hơn (a) vì tải và cắt chạy đan xen.
- Đỉnh đĩa (a) = tar + clip tích luỹ; (b) = j mp4 + clip tích luỹ. Nếu shard được upload ngay khi đầy thì phần clip giới hạn ở cỡ shard.
- Một shard đầy dở cộng một shard đang upload là ≤ 2 × 2 GB [ước lượng].

### 3.2 Bước lỗi [đo thực tế]

| Video | Trên Colab (Linux, decord 0.6.0 pip, ffmpeg/ffprobe 6.1.1 của hệ thống) | Trên Windows | Codec (ffprobe) | Bước |
| --- | --- | --- | --- | ---: |
| `5u97b1SqF1s`, `6-XNJpDWZCA`, `6LRqrQj0WXQ`, `6DGTIsQfGM4`, `6E6S3-BA41M`, `6ADVLgyyDlo` | `video_reader.cc:151 … cannot find video stream with wanted index: -1` | đọc được (`6DGTIsQfGM4`: 1 bước `past_end`, start 107.7 s ≥ dài 92.68 s) | **AV1** yuv420p | 45 |
| `5t3SrWk74L0`, `6FdIHkVlNoI` | `filter_graph.cc:61 … Cannot create buffer source` (pix_fmt −1) | cùng lỗi | h264, `pix_fmt` rỗng | 15 |

- Cùng một tar, 60/558 bước (10.8 %) và 8/82 video (9.8 %) không cắt được trên Linux.
- Loader train `data_qwen.py` cũng dùng decord, nên những video này cũng sẽ lỗi nếu train đọc mp4 trực tiếp. Vấn đề không riêng gì clip cắt sẵn.

### 3.3 Codec trên toàn bộ video cần dùng [đo thực tế]

Script: `scripts/probe_codecs_htstep.py`, chạy trên Colab. ffprobe đọc header mp4 qua giao thức `subfile` trên URL range của tar, không tải video. Video nào `pix_fmt` rỗng thì thử giải mã 1 khung bằng ffmpeg. Output: `tai_video/codecs.jsonl`.

7 620 video cần dùng; số bước lấy từ `items_*.jsonl`. Tóm tắt nằm ở `tai_video/codecs_summary.txt`.

| Codec (ffprobe) | decord 0.6.0 trên Linux | Video train | Bước train | Video val | Bước val |
| --- | --- | ---: | ---: | ---: | ---: |
| h264 yuv420p | đọc được | 5 828 | 35 652 | 180 | 1 039 |
| vp9 yuv420p | đọc được (thử `part1/-C_tbZixrmo.mp4`: OK) | 814 | 5 136 | 310 | 1 834 |
| **av1** yuv420p | **không** (tar 4007: 6/6 lỗi; thử `part1/-zJbtWpcdd4.mp4`: `cannot find video stream`) | **366** | **2 580** | **20** | **117** |
| **h264, `pix_fmt` rỗng, ffmpeg không giải mã được khung nào** ("stream received no packets": track video rỗng, file tải hỏng) | **không** (tar 4007: 2/2 lỗi) | **98** | **768** | 0 | 0 |
| h264, `pix_fmt` rỗng, nhưng giải mã được 1 khung | chưa thử | 2 | 8 | 0 | 0 |
| ffprobe lỗi mạng sau 3 lượt (`9W-KE40BxXA`, `PhnkxzBrmeE`) | chưa rõ | 2 | 7 | 0 | 0 |
| **Không đọc được (av1 + track rỗng)** | | **464 / 7 110 (6.5 %)** | **3 348 / 44 151 (7.6 %)** | **20 / 510 (3.9 %)** | **117 / 2 990 (3.9 %)** |

- Toàn bộ dữ liệu có khoảng **43 676** bước đọc được (47 141 − 3 465), chưa trừ `past_end`.
- Tar 4007 xấu hơn trung bình (10.8 % bước) vì có 6 video AV1.
- Lượt đầu, 143 video bị lỗi mạng (HTTP 5XX, TLS EOF khi chạy 24 luồng). Chạy lại ở 8 luồng thì còn 2.

### 3.4 Windows và Linux không cho cùng clip [đo thực tế]

- Trong 498 bước ok ở cả hai máy, **460 có cùng sha mảng, 38 khác** (25 video). Shape frame, `video_length`, độ dài wav và số byte file đều trùng.
- Thư viện lệch giữa hai máy: decord 0.6.0 cả hai bên nhưng wheel khác nhau (bản Windows giải mã được AV1); PyAV 17.1.0 (local) so với 19.0.2 (Colab).
- Chưa tách được khác ở frame hay ở wav, vì chỉ lưu sha chung [chưa xác minh].
- Hệ quả: clip phải cắt trên **Linux, cùng bản decord / av với máy sẽ train**. Kiểm tương đương cos ≥ 0.999 trong ô probe (plan mục 3) chạy trên máy GPU Linux sẽ bắt được lệch này. `training/requirements.txt` hiện không ghim `decord` / `av`, còn `run_vast.sh:38` cài bản mới nhất.

---

## 4. Ước lượng cho toàn bộ dữ liệu

### 4.1 Tổng [ước lượng]

Hệ số nhân từ tar 4007 lên toàn bộ:
- theo bước: 47 141 / 558 = **84.5**;
- theo byte: 183.70 / 1.971 = **93.2**.

Thời gian cắt tỉ lệ theo bước, thời gian tải tỉ lệ theo byte.

| Đại lượng | Giá trị | Cách tính |
| --- | ---: | --- |
| Byte tải, cách (b) | **183.70 GB** | mục 1.3 |
| Byte tải, cách (a) | 185.42 GB | tải mọi tar có ít nhất một member cần (bỏ tar 1001) + val repo: 179.55 − 1.69 + 7.56. Nếu tải luôn tar 1001: 187.1 GB |
| Giờ CPU cắt (Xeon 2.2 GHz của Colab) | **15.7 giờ vCPU** | 1.20 s/bước × 47 141 |
| Giờ CPU cắt (CPU local) | 7.1–8.3 giờ | 0.54–0.63 s/bước × 47 141 |
| Clip | **khoảng 41 GB** | 0.944 MB/bước ok × 43 676 bước đọc được (mục 3.3) = 41.2 GB. Nếu cứu được cả AV1 / file hỏng thì cận trên là 47 141 bước, 44.5 GB |
| Đỉnh đĩa, (b) | khoảng **4–5 GB** | j × mp4 lớn nhất (133 MB; p99 74 MB) + shard đầy dở ≤ 2 GB + shard đang upload ≤ 2 GB |
| Đỉnh đĩa, (a) | khoảng 7 GB mỗi tar đang xử lý | tar lớn nhất 2.58 GB + mp4 tạm + 2 shard. Theo `precut_htstep.fetch` hiện tại (giải nén hết rồi mới cắt): khoảng 2 × 2.58 GB mỗi job |

### 4.2 Theo nơi chạy [ước lượng, trừ các số đo ghi rõ]

| Nơi | Tài nguyên | Tiến trình | Thời gian toàn bộ | Chi phí | Nhận xét |
| --- | --- | ---: | --- | --- | --- |
| **Colab free (CPU)** | [đo thực tế] 2 vCPU, 12.7 GB RAM, đĩa `/` 108 GB trống (`df -h`), mạng HF 112 MB/s | 2 (j=4 chậm hơn, đo) | 426 s/tar × 84.5–93.2 = **10.0–11.0 giờ** | 0 | Phiên free tối đa khoảng 12 giờ và bị thu hồi khi idle [chưa xác minh: theo FAQ Colab; runtime tạo qua `colab` CLI từng bị thu hồi sau 10–25 phút idle, xem memory colab-cli-ssh]. Một phiên vừa sát giới hạn, nên chia **2 phiên / 2 tài khoản**, mỗi phiên một nửa danh sách, khoảng 5–5.5 giờ |
| **Máy local** | [đo thực tế] 8 nhân / 16 luồng, RAM trống 3.6 GB, D: trống 48 GB, mạng HF khoảng 4.2–4.5 MB/s | 3 (theo RAM: khoảng 1.2 GB/worker) | Giới hạn bởi mạng: 435 s/tar × 93.2 = **khoảng 11.3 giờ** | 0 | **Không dùng để tạo clip chính thức**: Windows cho clip khác Linux (mục 3.4). Clip khoảng 41 GB gần bằng 48 GB đĩa trống. Tốc độ upload lên HF chưa đo [chưa xác minh] |
| **Vast.ai** (máy GPU rẻ, chỉ dùng CPU) | ảnh chụp bảng giá 2026-11-02 15:43 (`tai_video/vast_offers.json`): 20 nhân / 63 GB / 811 Mbps **$0.082/h**; 28 nhân / 16 GB / 628 Mbps $0.096/h; 56 nhân / 94 GB / 935 Mbps $0.215/h. Phí băng thông $1.3–53 mỗi TB tuỳ máy | min(nhân, RAM GB / 2.5) (RAM đỉnh 2.2–2.3 GB/worker đo trên Linux): 20 với máy 20 nhân / 63 GB; chỉ 6 với máy 28 nhân / 16 GB | CPU: 56 600 vCPU-s / 20 = khoảng 47 phút. Mạng: 183.7 GB ở khoảng 80–100 MB/s = 31–38 phút, chạy đan xen. Upload khoảng 41 GB clip. Tổng **khoảng 1–2 giờ** | **$1–3**: giờ máy khoảng $0.2 + băng thông 0.23 TB × $4–13/TB | Giả định 1 vCPU của Vast nhanh ngang 1 vCPU Colab; chưa đo [chưa xác minh]. Bảng giá thay đổi liên tục |
| Song song nhiều máy | Mỗi máy một file danh sách đơn vị (tar / `part1` / `part2`) không trùng nhau, như `shards_part*.csv` của UniAV | | Wall = tổng / số máy, nếu chia đều theo **số bước** chứ không theo số tar (tar có 3–637 bước) | | Upload vào cùng repo HF: HF từ chối commit đồng thời, phải retry (đã có, `precut_htstep.py:77-90`). Tên shard có tiền tố máy để không trùng |

---

## 5. Thiết kế lưu và resume (mô tả, chưa viết đầy đủ)

**Đơn vị công việc là video**, không phải tar: một video = range-read một mp4, cắt mọi bước, xoá mp4.

1. **Danh sách**: `worklist_<máy>.txt`, mỗi dòng một `video_id`. Lập từ `items_*.jsonl` + `tar_index.csv`, chia theo số bước cho cân. Hai danh sách không được trùng nhau: kiểm khi lập.
2. **Trạng thái theo video**: mỗi video xong thì append một dòng vào `state/done.jsonl`, gồm `video_id`, các bước `{id, status, why, window, video_length, sha, bytes}` và tên shard chứa nó.
   - Chạy lại thì bỏ video đã có trong `done.jsonl` **và** đã nằm trong một shard đã upload. Video nằm trong shard chưa upload thì cắt lại.
   - Lỗi tải ghi vào `failed.jsonl` (như `failed.csv` của pseudocap), không làm dừng cả lượt.
3. **Shard**:
   - Clip ghi vào `pending/` rồi gom vào `shards/clips_<máy>_<NNNN>.tar` (member `clips/<video_id>/<step_idx>.npz`). Khi tổng clip ≥ 1.5 GB (trong khoảng 1–2 GB) thì đóng shard.
   - Mỗi shard có `clips_<máy>_<NNNN>.manifest.jsonl` (id, sha256 file npz, bytes, status) để kiểm toàn vẹn cuối và để máy train biết lấy item nào.
4. **Upload ngay khi shard đầy**:
   - Upload `.tar` trước, `manifest` sau. Có manifest trên HF nghĩa là shard đủ (cùng ý với `precut_htstep.py:143-144`).
   - Upload xong mới xoá shard local và đánh dấu video là `uploaded`.
   - Không ghi đè shard đầy hơn (như `upload_partial.py:17`): trước khi upload một shard trùng tên, đọc manifest trên HF; nếu bản trên HF có ≥ số item thì bỏ qua.
   - Shard đầy dở lúc phải dừng: upload với hậu tố `.partial` và ghi `partial.txt` (như `extract_features.py:272-297`). Lượt sau làm tiếp và thay bằng shard đủ.
5. **Nhiều phiên**: đánh dấu kiểu `claim_tar`, chuyển sang theo danh sách. Mỗi máy chỉ đụng danh sách của mình, nên không cần khoá chung. Trạng thái (`done.jsonl`, `failed.jsonl`) đẩy lên HF cùng manifest, để máy mới (đĩa trống) tải về và resume.
6. **Kiểm toàn vẹn cuối**:
   - Tải mọi manifest, so với `items_train.jsonl` / `items_val.jsonl`. Mỗi `id` phải có đúng một dòng: `ok` có sha256, hoặc `past_end` / `unreadable` có lý do.
   - In bảng item thiếu theo lý do: `missing` (chưa cắt), `fetch_error`, `unreadable`, `past_end`.
   - Cùng `id` xuất hiện ở hai shard là lỗi.
   - Kết quả này là đầu vào cho `convert_htstep.py finalize`.
7. **Máy train** (Colab / Vast) chỉ chạy `hf download <repo clip> --include "clips/*.tar" "*.manifest.jsonl"` rồi giải nén, **không tải mp4**. Kích thước: khoảng 41 GB clip.

---

## 6. Rủi ro và cách xử lý

| Rủi ro | Đã thấy? | Cách xử lý |
| --- | --- | --- |
| mp4 decord không đọc được (AV1, track video rỗng) | **Có**: 6.4 % video toàn bộ (484/7 620), 9.8 % ở tar 4007 (mục 3.2–3.3) | Ghi `unreadable` kèm lý do (đã có). Có transcode hay không: câu hỏi mở 1 |
| mp4 ngắn hơn nhãn | Có, 1 bước `past_end` (`6DGTIsQfGM4`, nhưng video này là AV1 nên trên Linux đã lỗi trước đó) | Kẹp `end` / bỏ `past_end` theo plan 2.2 (đã có trong `cut_clip`) |
| mp4 tải thiếu byte / hỏng | Không (0 lỗi tải; a = b từng mảng) | Kiểm `206` + đủ `size` (đã có trong `range_get`). Thêm: so sha256 của `.mp4` với lần tải khác thì không cần, vì size + cùng kết quả cắt là đủ |
| Rate limit / lỗi mạng HF | Không trong lần đo (24 luồng ffprobe + 4 luồng range) | Retry backoff `5·2**k`, tối đa 5 lần. Resolve URL CDN mỗi video (URL ký có hạn). Không gửi token cho CDN. Repo video là **public**, nên không cần token để đọc |
| Phiên Colab bị ngắt giữa chừng | Có thể (giới hạn khoảng 12 giờ, idle) | Resume theo video. Mất tối đa shard đầy dở (≤ 1.5 GB clip, khoảng 1 600 bước, khoảng 15–20 phút cắt ở j=2) và các video đang cắt dở |
| Upload lỗi giữa chừng | Không thử | Upload tar trước, manifest sau. Shard không có manifest coi như chưa có, upload lại. Xoá local chỉ sau khi `upload_file` trả về |
| Commit HF đồng thời từ nhiều máy | Đã gặp ở UniAV (lý do có retry ở `precut_htstep.py:77-90`) | Retry backoff, tên shard có tiền tố máy |
| Thư viện khác giữa máy cắt và máy train | **Có** (mục 3.4) | Cắt trên Linux; ghim `decord==0.6.0` và một bản `av` cố định cho cả precut và train (câu hỏi mở 2) |
| Đĩa đầy | Không (Colab trống 107 GB) | `need_disk` trước mỗi video (như pseudocap); (b) chỉ cần khoảng 4–5 GB |
| RAM | Linux khoảng 2.2 GB/worker | j ≤ RAM / 2.5 GB |

---

## 7. Các con số trong plan khác thực tế

Chỉ ghi lại, không sửa plan.

- Plan mục 3 viết "Tổng mp4 train cỡ vài chục đến gần 100GB". Thực tế mp4 train là **172.50 GB**, cả train và val cần tải là 183.70 GB [đo thực tế].
- Plan mục 3 viết precut "duyệt theo tar … resume theo tar". Báo cáo này đề xuất resume theo video và shard 1–2 GB (mục 5); `precut_htstep.py` hiện vẫn theo tar.

## 8. Câu hỏi mở (cần bạn quyết)

1. **Video AV1 (386) và mp4 track video rỗng (98).** decord 0.6.0 pip trên Linux không đọc được, mất 3 348 bước train (7.6 %) và 117 bước val (3.9 %).
   - Với AV1, các lựa chọn:
     - (i) bỏ như luật hiện tại (`unreadable` thì excluded);
     - (ii) transcode sang H.264 bằng ffmpeg rồi cắt (thêm CPU, frame qua hai lần nén);
     - (iii) dùng decoder khác cho riêng các video này (khi đó clip không còn "đúng như `video_decord` đọc mp4").
   - 98 file track rỗng chỉ có thể tải lại từ YouTube hoặc bỏ.
2. **Ghim thư viện**: ghim `decord` và `av` cho cả precut lẫn train / eval không? `training/requirements.txt` chưa ghim.
3. **Dung lượng clip khoảng 41 GB.** Nguyên nhân: frame uint8 nén zlib (`savez_compressed`) chỉ còn khoảng 0.94 MB mỗi bước. Lưu JPEG sẽ nhỏ hơn nhiều nhưng không còn pixel-exact. Giữ npz như plan?
4. **Nơi chạy**: 2 phiên Colab free (khoảng 5–5.5 giờ mỗi phiên, 0 đồng) hay 1 máy Vast (khoảng 1–2 giờ, khoảng $1–3)?
5. **Sửa `precut_htstep.py`** theo mục 5 (range theo video, shard 1–2 GB, resume theo video) trước khi chạy thật? Lượt này không sửa.

---

## Phụ lục

### A. Script (đều trong Omni-fix)

| Script | Việc |
| --- | --- |
| `scripts/inventory_htstep_videos.py` | Mục 1: HF API + header tar |
| `scripts/bench_precut_one_tar.py` | Mục 3: đo (a) / (b) trên một tar |
| `scripts/probe_codecs_htstep.py` | Mục 3.3: ffprobe qua range, chạy trên Linux |
| `data/htstep/tar_index.csv` | Bản sao chỉ mục tar (offset member), để (b) và probe chạy được trên máy không có `D:\Học\KL\Data` |

### B. Lệnh và output

```
# mục 1 (local, 18 s)
python -X utf8 scripts/inventory_htstep_videos.py --out docs/htstep/tai_video
  -> tai_video/inventory.out.txt, inventory.json, inventory_per_tar.csv

# mục 3, local (Windows; PYTHONPATH = transformers 4.51.3 như máy train)
python -X utf8 scripts/bench_precut_one_tar.py --processor <WAVE-7B preprocessor_config.json> \
    --work D:/htstep_bench --modes a b --jobs 3 --out docs/htstep/tai_video/bench_4007_local.json
  -> tai_video/bench_4007_local.{json,out.txt}

# mục 3, Colab free CPU (runtime tạo bằng `colab new -s htcpu`, code gửi qua `colab exec`, vì nhánh chưa push)
pip install -q "transformers==4.51.3" decord av "huggingface_hub[cli]" psutil
    # -> transformers 4.51.3, huggingface_hub 0.36.2, tokenizers 0.21.4, decord 0.6.0, av 19.0.2; Python 3.13
python -X utf8 scripts/bench_precut_one_tar.py --processor wave --work /content/bw --modes a b --jobs {2,4} \
    --out bench_4007_colab_j{2,4}.json
  -> tai_video/bench_4007_colab_j2.{json,out.txt}, bench_4007_colab_j4.{json,out.txt}
nproc; free -g; df -h /      # 2 | 12 GB (11 available) | overlay 108G, 108G trống

# mục 3.3, Colab
python scripts/probe_codecs_htstep.py --out codecs.jsonl --threads 24
  -> tai_video/codecs.jsonl (3 lượt: 24 luồng, rồi 2 lượt chạy lại các video lỗi ở 8 luồng; resume giữ dòng đã đọc được), codecs_summary.txt

# mục 4.2, Vast (API công khai, không cần khoá)
GET https://console.vast.ai/api/v0/bundles/?q={verified, rentable, rented=false, num_gpus>=1,
    cpu_cores_effective>=8, disk_space>=100, inet_down>=500, on-demand, order dph_total}
  -> tai_video/vast_offers.json (64 máy, 2026-11-02 15:43)
```

Trong các file `*.json` của bench có `per_video` (tải, cắt, RAM) và `per_step` (status, lý do, sha mảng, bytes, giây) cho từng video, từng bước.
