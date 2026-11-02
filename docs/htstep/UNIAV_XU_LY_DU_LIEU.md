# UniAV xử lý dữ liệu HT-Step thế nào (để Omni làm giống)

Ngày khảo sát: 2026-11-02. Chỉ đọc code UniAV, không sửa, không chạy model, không tải video / đặc trưng.
Nhãn: **[đọc code]** = đọc `file:dòng`; **[đếm thực tế]** = script trong `D:\Học\KL\Data\Htstep\khao_sat\` (phụ lục);
**[chưa xác minh]** = chưa kiểm được, ghi rõ vì sao. Đường dẫn code UniAV tính từ `Code/UniAV/UniAV-fixed`, nhánh
`htstep-artc` @ `8d688ff` trừ khi ghi khác.

## Tóm tắt (10 dòng)

1. Nhánh mô tả: `htstep-artc` @ `8d688ff` (local đang ở đúng commit này, sạch). Chưa xác định được checkpoint nào nhóm sẽ dùng (mục 11).
2. Nhãn: `annotations.json` + `taxonomy.csv` **gốc** của `facebookresearch/htstep`, không qua tiền xử lý nào; `anno/htstep.json` trên HF chỉ dùng ở các nhánh cũ.
3. Split: train = `subset == "train"` (17 472 video), đánh giá = `val_seen` (600 video). **Không có dev**; checkpoint chọn theo mAP trên chính val_seen.
4. Lọc bước: chỉ `end > start`, rồi lấy **48 bước đầu theo thứ tự trong json**. **Giữ cả `Full Match` lẫn `Partial Match`**, không sắp xếp, không gộp, không kẹp.
5. Sau lọc: train 103 258 bước (bỏ 35 bước của 1 video > 48 bước), val_seen 3 441 bước (không bỏ bước nào). Full Match chỉ chiếm 27 % / 29 %.
6. Video bị loại khi không có file `.npz`, file hỏng, hoặc mọi bước bắt đầu sau khi đặc trưng hết. Trên HF có `.npz` cho 4 813 train / 510 val_seen (báo cáo ghi dùng 4 810 / 505); cả 4 813 video train đều có ≥ 1 bước Full Match.
7. Thời gian: giây, như json. Lưới = 256 ô trên `T` = số hàng đặc trưng (≈ số giây của mp4), không dùng `duration` của json; giây ↔ ô: `g = t·256/T − 0.5`, `t = (g + 0.5)·T/256`.
8. Text của bước: `pool_text` = `headline.strip()` (giữ hoa thường, dấu câu); text đưa vào InternVideo2 = `"<activity>: <headline>"`, trừ mean rồi L2.
9. Id: `annotations[*].id` = `global_step_index`; hàng pool = thứ hạng của id trong taxonomy đã sắp (4 958 hàng, id 0..5784, 827 khoảng trống). Đầu ra `labels` là **hàng pool, không phải `global_step_index`**.
10. Omni: `omni_emb_file` (npz `ids` + `emb` 3584-d theo step) **chưa ai sinh cho HT-Step**; nếu có thì chỉ dùng trong loss lúc train. UniAV HT-Step không ghi file đầu ra nào; Fushion hiện chưa đọc đầu ra của nhánh này.

---

## 2. Nhánh nào là chuẩn

### 2.1 Trạng thái local [đếm thực tế]

```text
$ git branch -vv            (Code/UniAV/UniAV-fixed)
* htstep  8d688ff HT-Step article conditioning (artc), article head, mAP checkpoint selection
$ git status                -> nothing to commit, working tree clean
$ git log -5 --oneline
8d688ff HT-Step article conditioning (artc), article head, mAP checkpoint selection
99a8390 Step text vectors from htstep-text-feats; drop dim notes at the top of event_archs.py
742f302 han li
cd5ebda HT-Step minimal: minimal layout on HT-Step k768 features, DDP via torchrun, Python 3.13 / torch 2.14
3f8cd35 fix
```

Nhánh local `htstep` không có upstream, nhưng trỏ đúng commit của `origin/htstep-artc` (sau `git fetch --all`). Nó **không** phải
`origin/htstep` (`13114d8`). `remote origin` = `github.com/im-xiaoming/UniAV-fixed` (request ghi `HanLi-Retriever`; không kiểm được là
cùng repo đổi tên hay khác repo).

### 2.2 Hai dòng nhánh HT-Step [đọc code]

| Nhánh | Commit, ngày | Nhãn đọc từ | `partial` | Text bước | Tối đa bước | Đặc trưng |
|---|---|---|---|---|---|---|
| `htstep` | `13114d8`, 10-21 | HF `htstep-feats/anno/htstep.json` → `data/htstep/annotations.json` (định dạng YouCook2) | chỉ `Full Match` | `headline.strip()` (từng caption) | 16 | `v768` stage2 (`configs/htstep_event.yaml`) |
| `pipeline-k400` | `bf129b1`, 10-21 | như `htstep` (chỉ thêm `tools/pipeline/`) | Full | headline | 16 | như trên |
| `cleanup-htstep` | `7c415ac`, 10-29 | như `htstep` | Full | headline | 16 | như trên, cho phép `k768` |
| `htstep-minimal` | `cd5ebda`, 10-30 | HF `anno/htstep.json` | Full | headline | 16 | `k768` |
| `htstep-ov` | `742f302`, 10-30 | `annotations.json` + `taxonomy.csv` gốc | **Full + Partial** | `"<activity>: <headline>"` (4 958 step) | 48 | `k768` |
| **`htstep-artc`** | **`8d688ff`, 11-01** | như `htstep-ov` | Full + Partial | như `htstep-ov` | 48 | `k768` |
| `chien` | `cd11ee8`, 10-23 | không có HT-Step (chỉ YouCook2) | – | – | – | – |

Hai dòng không chung gốc gần: `htstep → pipeline-k400 → cleanup-htstep` (merge-base `13114d8`) và
`minimal → htstep-minimal → htstep-ov → htstep-artc`. `docs/DATA_PIPELINE.md` chỉ có ở `origin/main` (2026-10-03, UniAV gốc
DESED/UnAV, không nói HT-Step).

### 2.3 Nhánh nào train bản Athena HT-Step nhóm sẽ dùng

- `experiments_report.md:1-3` (trên `htstep-artc`) ghi các lần chạy 31/10–01/11 "Nhánh: `htstep-ov`", log `logs/train_<tên>.log`,
  checkpoint `ckpt/htstep_<tên>/model_best.pth.tar`. Không có log hay checkpoint HT-Step nào trong repo local (`ckpt/` chỉ có
  `athena.pth`, `athena_coin.pth`, `model_epoch_006.pth.tar`, đều YouCook2/COIN theo `ckpt/README.md`).
- Kết quả tốt nhất ghi trong báo cáo (`experiments/README.md:26-34`): artc 30.32, artc + `max_seq_len` 512 = 30.62 mAP. Nhưng config mặc định
  của `htstep-artc` có `article_cond: False`, `max_seq_len: 256` (`libs/core/config.py:22,34`), tức là bản baseline 25.70.
- `experiments/runs/*.json` (3 file) đều là lần chạy thử 0 epoch trên `htstep-ov` `cd5ebda`, dirty.
- **Kết luận:** mục 3–10 mô tả `htstep-artc` (local, mới nhất, chứa đủ code của cả baseline lẫn artc). Bản checkpoint / cấu hình cụ thể: **không
  xác định được** → mục 11.

## 3. Nguồn nhãn

- [đọc code] `json_file = ./data/htstep/data/annotations.json`, `taxonomy_file = ./data/htstep/data/taxonomy.csv` (`libs/core/config.py:12-13`),
  là repo gốc `facebookresearch/htstep` clone nguyên bởi `tools/htstep_data.py fetch` (`tools/htstep_data.py:71-72`). Dataset đọc thẳng
  (`libs/datasets/htstep.py:66-67,100-101`). **Không có bước tiền xử lý nào ghi ra file nhãn mới.**
- [đọc code] `anno/htstep.json` (HF `nguyenminh04/htstep-feats`, **không phải** `htstep-feats-k400`) chỉ được đọc ở nhánh cũ:
  `origin/htstep:tools/htstep_data.py:37-41` và `origin/htstep-minimal:tools/htstep_data.py:44`. File này do
  `origin/htstep:tools/htstep_hub.py:117-122` sinh: giữ bước `partial == 'Full Match'` có id trong taxonomy, caption = headline.
- [đếm thực tế] `anno/htstep.json` = 12 023 video (11 618 train + 405 val_seen), 28 770 bước = **đúng toàn bộ bước `Full Match`** của
  `annotations.json`; 12 023/12 023 video trùng khít (mốc, thứ tự, headline taxonomy) với bản gốc lọc Full Match.
- [đếm thực tế] `annotations.json` local **đã bị sửa** so với HEAD của repo gốc (`git status`: `M data/annotations.json`, mtime 2026-11-01):
  cùng 18 072 video, cùng mọi trường, chỉ **thêm trường `headline`** vào mỗi bước (106 734 bước), và `headline` này trùng 100 % với
  headline taxonomy của `id`. UniAV không đọc trường này (lấy text từ taxonomy, `htstep.py:71`), nên kết quả UniAV như nhau với hai bản.
  `taxonomy.csv` local trùng HEAD (khác nhau chỉ CRLF, sha256 sau khi bỏ `\r` bằng nhau).

## 4. Tập video và chia split

### 4.1 Giá trị `subset` [đếm thực tế]

| Nguồn | train | val_seen | khác |
|---|---|---|---|
| `annotations.json` (`subset`) | 17 472 | 600 | không có giá trị nào khác |
| `video_splits.json` | 17 472 | 600 | `test_seen` 600, `test_unseen` 1 000, `val_unseen` 963 (không có trong `annotations.json`) |

[đọc code] Train = `['train']`, đánh giá = `['val_seen']` (`train.py:52-53`); so sánh `value['subset'].lower() not in self.split`
(`htstep.py:104`).

### 4.2 Dev

[đọc code] **Không có dev.** Không có tách theo video / article / activity, không seed, không file danh sách. Checkpoint chọn theo mAP
val_seen (`train.py:103-122`). (Ngoài UniAV có `Data/Htstep/pseudocap/metadata/htstep_dev_v2.json`, 72 video, của việc pseudo-caption;
UniAV không đọc.)

### 4.3 Video bị loại [đọc code `htstep.py:103-119`, theo thứ tự]

1. `subset` không thuộc split (104);
2. không còn bước sau lọc (106-108) — thực tế 0 video;
3. không có `<feat_folder>/<id>.npz` (109-111);
4. file hỏng: `np.load` lỗi `OSError/ValueError/KeyError/BadZipFile` khi đọc `k768` và `a768` (112-116); file thiếu khoá `k768` cũng rơi vào đây;
5. đặc trưng ngắn: bước bắt đầu sớm nhất `>= rows` với `rows = min(len(k768), len(a768))` (117-119).

Không lọc trùng YouCook2 (không có điều kiện nào). [đếm thực tế] Trùng id với `Data/YouCookII/metadata/youcookii_annotations_uemr.json`
(1 790 video YC2): train 1 (`6MBctYaMU8U`), val_seen 0.

Đặc trưng được `tools/htstep_data.py fetch` lấy về: mọi `.npz` size > 0 có id trong `annotations.json`, từ mọi tar của
`htstep-feats-k400` rồi `htstep-val-seen-feats`, file sau ghi đè file trước (`tools/htstep_data.py:75-93`). Không kiểm split.

| Đếm | train | val_seen |
|---|---|---|
| Video trong `annotations.json` [đếm thực tế] | 17 472 | 600 |
| Có `.npz` size > 0 trên HF (cận trên, trước bước 4–5) [đếm thực tế] | 4 813 | 510 |
| — trong `htstep-feats-k400` / `htstep-val-seen-feats` | 4 813 / 0 | 179 / 331 |
| — có ≥ 1 bước Full Match | **4 813 (100 %)** | 348 |
| Có mp4 trên HF (tar `htstep-videos` ∪ `htstep-val-seen-videos`) [đếm thực tế] | 7 110 | 510 |
| Có `.npz` nhưng không thấy mp4 trong repo video nào [đếm thực tế] | 361 | 0 |
| UniAV dùng thực tế (báo cáo, `experiments_report.md:12-16`) [chưa xác minh] | 4 810 | 505 |

[đếm thực tế, `so_lieu_dac_trung.json`; tên file trong tar đọc qua header, không tải nội dung]

- Train: mọi video có đặc trưng đều có Full Match, vì danh sách tải / trích xuất lấy từ `anno/htstep.json` (chỉ video có Full Match,
  `origin/htstep:tools/htstep_hub.py:117-122`). Tức là **tập train thực tế = video có ≥ 1 Full Match**, nhưng dùng **mọi** bước (cả Partial)
  của các video đó. Chênh 4 813 − 4 810 = 3 [chưa xác minh: hỏng / ngắn, hay báo cáo chạy trước khi có đủ tar].
- val_seen: 510 có `.npz` = đúng 510 video có mp4. Nguồn file cuối (theo thứ tự `fetch`): 179 từ tar chunk của `htstep-feats-k400`,
  `part1_pc` 103, `val_iv2` 78, `part1_colab` 66, `part2_vast` 58, `part2_pc` 17, `part2_colab` 9. Chênh 510 − 505 = 5 khớp "5 video có file npz
  bị hỏng" (`experiments_report.md:13-15`, `broken_feats.txt` không có trong git) [chưa xác minh từng id].
- 0 `.npz` rỗng; 484 id nằm ở > 1 tar (README `htstep-feats-k400`: "the files are identical") [chưa xác minh nội dung].
  `extracted_videos.txt` (4 856 id) thiếu 136 id có trong tar: file cũ.
- mp4: 361 video train có `.npz` nhưng không có trong 113 tar `htstep-videos` [chưa xác minh: có thể là `iv2_train.tar`, "705 more train videos
  extracted separately"]. 162 video val_seen có mp4 ở **cả** tar `htstep-videos` lẫn `htstep-val-seen-videos` (hai bản tải khác nhau).
  `htstep-videos/manifest.json` đã cũ (95 / 113 tar, 5 907 id); số trên dùng chỉ mục header 113 tar `Data/Htstep/pseudocap/metadata/tar_index.csv`
  (7 453 id).

### 4.4 Video không có audio

- [đọc code] Lúc trích: `a768` = mảng 0 khi không có model audio hoặc ffmpeg không ra mẫu nào (`len(wav) == 0`)
  (`tools/pipeline/extract_features.py:363-364`). Video có âm thanh câm thì vẫn có vector BEATs (không phải 0).
- [đọc code] UniAV **không phát hiện, không loại, không đánh dấu** video không audio: chỉ L2-chuẩn hoá từng hàng với mẫu số
  `max(norm, 1e-6)`, nên hàng 0 vẫn là 0 (`htstep.py:141-146`).
- [chưa xác minh] Số video không audio: cần mở `a768` của từng `.npz` (không làm theo quy tắc "không tải đặc trưng"). Liên quan: repo
  `htstep-val-seen-videos/part1/` còn 2 file `.f140.m4a` (`G_6gAP7wMAs`, `R--fZfXV_jg`) và 1 file `.f605.mp4` = mảnh yt-dlp chưa ghép; cả hai
  id vẫn có `part1/<id>.mp4` [đếm thực tế]; mp4 đó có audio hay không: [chưa xác minh].

## 5. Lọc bước

[đọc code `htstep.py:106`, đúng thứ tự]

```python
anns = [a for a in value['annotations'] if a['segment'][1] > a['segment'][0]][:self.num_classes]   # num_classes = 48
```

| Điều kiện | UniAV `htstep-artc` |
|---|---|
| Lọc `partial` | **không** (giữ Full Match và Partial Match) |
| `end > start` | có, trước khi cắt 48 |
| Giới hạn | 48 bước đầu **theo thứ tự trong json** (không sắp xếp) |
| Sắp xếp / gộp chồng lấn / kẹp vào `duration` | không / không / không (lúc đọc json) |
| Train khác đánh giá? | Lọc json giống nhau. Riêng train, trong `__getitem__` còn bỏ bước bắt đầu sau `256.5` ô, giữ bước bị cắt cuối nếu ≥ 50 % còn lại, kẹp `end ≤ 256.5` ô, luôn giữ ≥ 1 bước (`htstep.py:164-170`), rồi random crop (mục 6.4). Đánh giá dùng mốc giây gốc `segments_sec` (`htstep.py:203`). |

[đếm thực tế] (trước lọc / sau lọc, theo split)

| | train | val_seen |
|---|---|---|
| Bước trong json | 103 293 | 3 441 |
| — Full Match / Partial Match | 27 769 / 75 524 | 1 001 / 2 440 |
| Bỏ vì `end <= start` | 0 | 0 |
| Video > 48 bước / bước bị bỏ | 1 / 35 | 0 / 0 |
| **Bước sau lọc** | **103 258** | **3 441** |
| — trong đó Full Match | 27 769 | 1 001 |
| Bước / video (min, trung vị, p90, max) | 1, 5, 11, 48 | 1, 5, 11, 27 |
| Video có bước không theo thứ tự `start` | 3 880 | 144 |
| Cặp bước chồng lấn (sau khi sắp) / số video có | 1 325 / 1 052 | 42 / 37 |
| Bước có `end > duration` (json) | 76 | 3 |
| Bước có `start >= duration` hoặc `start < 0` | 0 | 0 |
| Bước trùng hệt (cùng id, mốc lệch ≤ 1e-3) | 0 | 0 |
| Video có ≥ 1 bước Full Match | 11 618 | 405 |

Nhánh cũ (`htstep`, `cleanup-htstep`, `pipeline-k400`, `htstep-minimal`): chỉ `Full Match`, kẹp `[0, duration]`, bỏ `end <= s` và caption rỗng
(`origin/htstep:tools/htstep_data.py:52-55`), tối đa 16 bước (`origin/htstep:libs/datasets/step_captions.py:27,117`).

## 6. Thời gian

### 6.1 Đơn vị

- [đọc code] `segment` là giây, đọc nguyên (`htstep.py:117`). `duration` lấy từ json (`htstep.py:123`), chỉ dùng để **kẹp đầu ra** lúc
  inference (`libs/modeling/event_archs.py:560`). Độ dài lưới lấy từ **số hàng đặc trưng** `T = min(len(k768), len(a768))` (`htstep.py:147`).
- [đếm thực tế] `duration` của json là số nguyên ở 100 % video (17 472 + 600). Train: 28–1 957 s, trung vị 342; val_seen: 37–1 759 s,
  trung vị 331.

### 6.2 Giây ↔ ô lưới [đọc code `htstep.py:149-162`, `event_archs.py:558-560`]

```text
T             = min(rows k768, rows a768)             # 1 hàng / giây
feat_stride   = T * default_fps / max_seq_len = T*16/256   # default_fps = 16 chỉ là hằng quy đổi, không phải fps thật
đặc trưng     : F.interpolate(T hàng -> 256 ô, mode='linear', align_corners=False)
giây -> ô     : g = t * 16 / feat_stride - 0.5 = t * 256 / T - 0.5          (feat_offset = 0.5)
ô -> giây     : t = (g * feat_stride + 0.5 * feat_stride) / 16 = (g + 0.5) * T / 256,  kẹp [0, duration_json]
```

- Ô `i` có tâm ở `(i + 0.5)·T/256` giây (lệch nửa ô có chủ đích, `feat_offset = 0.5`). Không làm tròn.
- Pyramid 6 tầng, stride `[1, 2, 4, 8, 16, 32]` ô (`htstep.py:86`); `regression_range` `[[0,4],[4,8],[8,16],[16,32],[32,64],[64,10000]]` ô
  (`config.py:47`). `max_seq_len = 256` (`config.py:22,52`). Video ngắn hơn 256 s bị kéo giãn, dài hơn bị nén.
- Fushion dùng cùng công thức `t = (g + 0.5)·n/256` (`Code/Fushion/uniav_app/detect.py:7`).

### 6.3 Đặc trưng k400 được trích từ đâu

- [đọc code] mp4 lấy từ `nguyenminh04/htstep-videos` (`tools/pipeline/download_hf.py:7-10,37`) hoặc YouTube bằng yt-dlp
  `bv*[height<=360][ext=mp4]+ba[ext=m4a]/b[height<=360]/bv*+ba/b`, ghép mp4 (`tools/pipeline/download_videos.py:49,114-115`). Không cắt
  đoạn (không `--download-sections`), không đổi fps trước khi trích.
- [đọc code] Trích: `ffmpeg -vf fps=4,scale=224:224` (`extract_features.py:212`); `n_sec = ceil(số khung / 4)` (225); hàng `i` dùng 16 khung
  `4i − 7 … 4i + 8`, tức cửa sổ `[i − 1.75, i + 2.0]` s, tâm `i + 0.125` s (226-227, mép lặp khung đầu/cuối). Audio 16 kHz mono, hàng `i` =
  BEATs trung bình trên `[i − 1, i + 2)` s, tâm `i + 0.5` (241-242). Model `1B_ft_k710_ft_k400_f16.pth`, khoá `k768` (85, 171).
- [đọc code] Lưới UniAV coi hàng `i` có tâm `i + 0.5` s khi `T = 256`; cửa sổ video thật có tâm `i + 0.125` s (chênh 0.375 s). Chỉ ghi sự thật.
- [chưa xác minh] Đặc trưng val_seen ở `htstep-val-seen-feats` (`part1_*`, `part2_*`, `val_iv2.tar`): README của repo ghi "same extractor and
  format"; mp4 nguồn của `part1/part2` là `htstep-val-seen-videos` (README repo đó), `val_iv2.tar` "extracted separately" không ghi nguồn
  video. Script tải `part2` không nằm trong UniAV (có `D:\Học\KL\dl_missing.py`: yt-dlp ≤ 360p, ưu tiên H.264 + m4a, không cắt).
- **Kết luận cho mốc thời gian:** nhãn là giây trên video YouTube gốc; mp4 HF là bản tải lại nguyên video (≤ 360p), không cắt, nên cùng trục
  thời gian với nhãn **nếu** bản tải trùng bản lúc HT-Step gán nhãn [chưa xác minh: không có mp4 local để so `duration`, mục 11].

### 6.4 Random crop lúc train (chỉ để biết) [đọc code `data_utils.py:22-95`]

Cửa sổ dài `randint(round(0.9·256), 256)` ô; trùng 256 thì không crop. Bước giữ lại nếu phần giao với `[st − 0.5, ed + 0.5]` ≥ 50 % độ dài
bước (`trunc_thresh = 0.5`); mốc bị kẹp vào cửa sổ rồi trừ `st`; thử tối đa 200 lần đến khi còn ≥ 1 bước.

### 6.5 Phân bố độ dài bước sau lọc (giây) [đếm thực tế]

| Tập | n | min | p10 | trung vị | p90 | p99 | max | > 8 s | > 30 s | > 60 s |
|---|---|---|---|---|---|---|---|---|---|---|
| train (UniAV) | 103 258 | 0.00 | 1.81 | 7.47 | 34.82 | 110.09 | 800.33 | 47.69 % | 12.42 % | 3.86 % |
| val_seen (UniAV) | 3 441 | 0.01 | 1.91 | 7.69 | 36.49 | 119.73 | 469.78 | 48.68 % | 12.79 % | 4.36 % |
| train, chỉ Full Match | 27 769 | 0.00 | 2.60 | 10.67 | 45.53 | 135.75 | 800.33 | 59.77 % | 18.27 % | 6.11 % |
| val_seen, chỉ Full Match | 1 001 | 0.17 | 2.74 | 10.76 | 45.14 | 147.44 | 398.59 | 60.94 % | 18.18 % | 5.79 % |

(min 0.00 = dương nhưng < 0.005 s.)

## 7. Text của bước

1. [đọc code] `pool_text = [r['headline'].strip() for r in rows]` theo taxonomy đã sắp (`htstep.py:71`): chỉ `strip`, **giữ** chữ hoa và dấu
   câu. Text GT của segment `cap_text = pool_text[step]` (`htstep.py:201`). [đếm thực tế] 0 headline có khoảng trắng đầu/cuối.
2. [đọc code] Text vào encoder: `'%s: %s' % (activity, headline.strip())` (`tools/htstep_data.py:48`). Encoder `InternVideo2TextEncoder`
   (`tools/iv2_text.py`): checkpoint `ckpt/internvideo2/InternVideo2-stage2_1b-224p-f4.pt` (`htstep_data.py:50-51`), BERT-large 19 tầng đầu,
   token `[CLS]`, `text_proj` 1024 → 512, L2; tokenizer `google-bert/bert-large-uncased`, text `strip().lower()`, cắt 40 token, fp32
   (`iv2_text.py:15-16,22,47-50`).
3. [đọc code] Hậu xử lý: `mean = raw.mean(0)` trên **cả 4 958 step**, `emb = raw − mean`, L2 lại (`htstep_data.py:53-55`). Lưu
   `step_emb_iv2j.npz`: `ids` (4958,) int = `global_step_index` đã sắp, `emb` (4958, 512) **float16**, `text` (4958,) str, `mean` (512,)
   float32 (`htstep_data.py:57-58`). Mean lưu trong chính npz. File lấy từ HF `nguyenminh04/htstep-text-feats/text_feats/step_emb_iv2j.npz`
   (`htstep_data.py:27,73-74`; 4 811 280 byte, commit 2026-10-31). README HF ghi cùng khoá/shape và "made on branch `htstep-ov`".
   [chưa xác minh] chưa mở npz (không tải đặc trưng). Trong model, vector còn qua `clip_proj` Linear 512→512 + L2 (`event_archs.py:113-117`).
4. [đọc code] `annotations[*].id` → `step_row[id]` = vị trí của id trong taxonomy sắp theo `global_step_index` (`htstep.py:66-70,120`).
   `load_step_vectors` đặt `emb[step_row[ids[k]]] = emb_k` (`htstep.py:17-25`). [đếm thực tế] 4 958 step, id 0..5784, 827 id trống, không id
   trùng; 0 bước của json có id ngoài taxonomy; train dùng 4 531 id khác nhau, val_seen 983.
5. Headline trùng [đếm thực tế]

   | | số |
   |---|---|
   | step trong taxonomy | 4 958 |
   | headline khác nhau (`strip`) | 4 594 |
   | `"<activity>: <headline>"` khác nhau | **4 872** (86 step trùng cả activity lẫn headline, khác variation → vector giống hệt) |
   | headline khác nhau sau chữ thường + bỏ dấu câu | 4 582 |
   | step có headline dùng chung với step khác | 501 |
   | bước val_seen có headline xuất hiện ở video val_seen khác | 86.40 % (cùng `id`: 85.73 %) |
   | bước train có headline xuất hiện ở video train khác | 99.44 % |

   Top 20 headline trùng (số step taxonomy): Finished. 58 · Serve. 56 · Serve warm. 19 · Heat the oil. 18 · Serve hot. 12 · Serve immediately. 7 ·
   Knead the dough. 7 · Roll out the dough. 6 · Boil the water. 6 · Gather your ingredients. 6 · Combine the dry ingredients. 6 · Preheat your
   oven. 5 · Chop the vegetables. 5 · Remove from the heat. 5 · Heat the oil in a skillet. 5 · Combine the ingredients. 5 · Roll the dough into
   balls. 4 · Add the remaining ingredients. 4 · Serve and enjoy. 4 · Mix well. 4.
6. [đọc code] Pseudo-caption / `athena/captioner.py` / `caption_pool.npz`: **không có** trên `htstep-artc`, `htstep-ov`, `cleanup-htstep`
   (`git ls-tree`: 0 file). Trên `origin/htstep` có `athena/` nhưng là Athena API của YouCook2 (8 218 câu train YC2, `athena/README.md:4`),
   không dùng cho HT-Step. Text phụ khác: `paragraph` của taxonomy không được đọc.

## 8. Nhãn và candidate

- [đọc code] Article = `(activity, variation)` của taxonomy (`htstep.py:75-80`); [đếm thực tế] 617 article (433 activity; `variation` gồm
  `none`, A–F), 1–27 step/article, trung vị 7.
- [đọc code] Article của video = article của **bước đầu tiên** (`htstep.py:129-130`). [đếm thực tế] mọi bước của mọi video cùng một article, và
  article đó trùng `(activity, variation)` của video trong json (0 lệch, 0 video nhiều article).
- [đọc code] `cand` = hàng pool của mọi step trong article (`htstep.py:129,197`). [đếm thực tế] cỡ `cand` mỗi video: train 1–27 (trung vị 9),
  val_seen 2–27 (trung vị 9).
- [đọc code] `labels = arange(n)` = **chỉ số segment trong video**, không phải lớp (`htstep.py:125`); `steps` = hàng pool của từng segment
  (`htstep.py:120`); `num_classes = 48` = số segment tối đa mỗi video (`config.py:25`), dùng làm chiều one-hot của `label_points`
  (`data_utils.py:162`). Lớp thật là 4 958 step, định nghĩa bằng text (open vocabulary).
- [đọc code] Loss dùng text vector của bước: `emb_loss` = CE giữa vector từng điểm dương và **cả 4 958** vector text đã chiếu, đích = hàng pool
  của segment chứa điểm (`event_archs.py:467-474`); `span_loss` tương tự cho vector trung bình segment (`SegmentContext`, 484-491);
  `omni_emb_loss` nếu có file Omni (476-481); `art_loss` nếu bật `article_head` (493-496). `article_cond` cho đặc trưng attention sang vector text
  (chưa chiếu) của các step trong article (`event_archs.py:398-403`). Lúc inference step của segment chọn trong `cand` (`test_cfg.article = True`,
  `event_archs.py:554-556`).

## 9. Giao diện với Omni

### 9.1 `omni_emb_file`

- [đọc code] Mặc định `''` → `omni_head_dim = 0`, không có head Omni (`config.py:17,46`; `train.py:45-46`).
- [đọc code] Định dạng đọc được: npz có `ids` (= `global_step_index`) và `emb` (N, D) (`htstep.py:17-25,73`), "same layout" như
  `step_emb_iv2j.npz` (`config.py:16`, `README.md:80-81`). D phải bằng `omni_head_dim = 3584` (`config.py:46`, conv ra 3584 kênh
  `event_archs.py:86-87`). Khoá `text` / `mean` không được đọc. Id thiếu → hàng 0 (`np.zeros`, `htstep.py:23`).
- [đọc code] UniAV **không trừ mean**; chỉ L2 trong loss: `pool_o = F.normalize(self.omni_pool_raw)` (`event_archs.py:480`). Vector Omni **không
  bị chiếu**; đặc trưng video được chiếu sang 3584-d (`omni_vis_proj`, MaskedConv1D, L2, `event_archs.py:84-87,108-110`).
- [đọc code] Dùng ở: **chỉ loss lúc train**, `omni_emb_loss = CE(scale · omni_em[điểm dương] @ pool_o.T, hàng pool GT)`, trọng số 0.2
  (`event_archs.py:476-481`, `config.py:59`); `scale` dùng chung với head InternVideo2. Inference không nhận `omni_em` (`event_archs.py:426`).
- Ai sinh file: **không có**. `git grep` `omni_emb|omni_pool|3584|OmniRetriever` trên mọi nhánh HT-Step: chỉ có chỗ đọc; nhánh cũ chỉ có
  `omni_emb_full.npz` của **YouCook2** (`experiments/RESULTS.md` các nhánh `htstep`/`cleanup-htstep`). Text đưa vào Omni: code không quy định.

### 9.2 Align vector sự kiện UniAV ↔ vector Omni

- [đọc code] Trên `htstep-artc`, align duy nhất là `omni_emb_loss` ở 9.1: đầu vào Omni **theo step** (`ids` = `global_step_index` → hàng pool), không
  theo video hay segment. Không có teacher Omni theo clip.
- [đọc code] Nhánh cũ `origin/htstep`: `libs/` không còn chỗ nào dùng Omni; chỉ `tools/export_api_ckpt.py:50,68` đọc `omni_emb_file` (YouCook2) và
  bỏ trọng số `omni_*` khi xuất checkpoint.

### 9.3 Đầu ra của UniAV lúc inference

- [đọc code] `inference()` trả list dict, mỗi video một dict (`event_archs.py:561-564`):
  `video_id`; `segments` (K, 2) **giây**, kẹp `[0, duration_json]`, xếp theo điểm sau soft-NMS, K ≤ 200; `scores` (K,); `labels` (K,) = **hàng
  pool** (chọn trong `cand`); `label_probs` (K,); `article_pred` (-1 nếu không có head); `gt_segments` (giây gốc); `gt_text` (headline).
  **Không có vector sự kiện `z_i`** (span vector tính ở dòng 550 nhưng không trả ra).
- [đọc code] Không ghi file: `valid_one_epoch` dùng trực tiếp trong bộ nhớ để tính độ đo (`libs/utils/train_utils.py:199-221`); `train.py --eval`
  chỉ in số (`train.py:80-86`). Checkpoint lưu `{'epoch', 'state_dict', 'metrics'}` (`train.py:121`).
- **Cập nhật 2026-11-02:** code HT-Step 0-shot của Fushion (`uemr_htstep/`, `configs/htstep_0shot_protocol.json`, notebook
  `uemr_htstep_0shot_AZ.ipynb`, test) đã bỏ ở commit `588bd38` (không làm nữa). Các trích dẫn tới những file đó bên dưới và ở mục 11 là
  theo commit `4c6dbb6`. `uniav_app/` và `uemr_fusion/` vẫn còn.
- [đọc code] Fushion (nhánh `htstep` @ `4c6dbb6`), chỗ nhận đầu ra "UniAV": hiện là Athena YouCook2 vendored từ UniAV-fixed nhánh `ov-refine`
  (`third_party/athena/VENDORED.md:3`), chạy qua `uniav_app`:
  - `raw_proposals.jsonl`: `{video_id, split, duration, n, feature_source, run, time_params, proposals: [{g_s, g_e, ts, te, conf}]}`, giây,
    `t = (g + 0.5)·n/256` kẹp (`uniav_app/detect.py:1-11,28-37,63-67`);
  - `Z/<video_id>.npz`: `z` fp16, `t_sec` float32, `meta` json (`uniav_app/z_export.py:32-41`);
  - Omni đọc lại theo `segments.npz` (`partition`, `video_id`, `ts`, `te`, `emb`) (`uemr_fusion/build_cache.py:61-73`).
  - Phía HT-Step của Fushion đọc `anno/htstep.json` (chỉ Full Match, headline trần), id câu hỏi `"<video_id>#<i>"` với `i` = vị trí trong danh
    sách Full Match của `htstep.json` (`uemr_htstep/data.py:5-6,67-77`), kẹp mốc theo mp4 (`data.py:10-12,80-93`). Đặc trưng là `stage2_f4`
    `v768`, không phải `k768` (`configs/htstep_0shot_protocol.json:425-426`).
  - **Không có** chỗ nào trong Fushion đọc đầu ra / checkpoint của `htstep-artc` (`git grep -E "model_best|htstep-artc|k768|label_probs|step_emb_iv2j|global_step_index"`
    trên nhánh `htstep`, bỏ notebook: chỉ 2 dòng, protocol ghi `k768` là "not_used" và `uemr_htstep/data.py:121` đối chiếu headline với
    taxonomy; phụ lục B.1).

### 9.4 Khoá chung giữa hai bên

| Khoá | Dạng chính xác ở UniAV `htstep-artc` | Nguồn |
|---|---|---|
| video id | YouTube id 11 ký tự, = key của `annotations.json` = tên `<id>.npz` | `htstep.py:103,109` |
| split | `subset` của `annotations.json`: `"train"` / `"val_seen"` | `htstep.py:104`, `train.py:52-53` |
| step id | `global_step_index` (int, 0..5784, có khoảng trống) = `annotations[*].id` = `ids` trong npz text | `htstep.py:17-25,120` |
| hàng pool | 0..4957 = thứ hạng `global_step_index` trong taxonomy đã sắp; là giá trị `steps`, `cand`, `labels` đầu ra | `htstep.py:66-70` |
| chỉ số bước trong video | vị trí trong `annotations` sau lọc `end > start`, cắt 48 (`labels`) — **không** phải vị trí trong `htstep.json` | `htstep.py:106,125` |
| article | `(activity, variation)` của taxonomy; chỉ số = thứ tự xuất hiện đầu tiên trong taxonomy đã sắp (617) | `htstep.py:75-80` |
| mốc thời gian | giây float như json (GT), đầu ra giây kẹp `[0, duration_json]` | `htstep.py:203`, `event_archs.py:560` |
| text bước | GT: `headline.strip()`; vector: `"<activity>: <headline>"` | `htstep.py:71`, `htstep_data.py:48` |

## 10. Đánh giá của UniAV (ngắn)

[đọc code `train_utils.py:141-231`] Mỗi epoch, trên val_seen (EMA model): R@0.3/0.5/0.7 và mIoU (top-k = số GT của video), CIDEr của headline
bước dự đoán trên GT khớp IoU ≥ 0.3, Acc (đúng step), mAP article grounding (tIoU `linspace(0.3, 0.7, 5)`, AP theo activity, bỏ dự đoán có
step ngoài article hoặc không là GT của video nào; VOC AP). GT gồm cả Full + Partial. Checkpoint = mAP tốt nhất (`train.py:118-122`); dừng sớm
khi mAP và CIDEr không tăng 3 epoch (`train.py:129-130`). Báo cáo ghi mAP này khớp script chính thức trên 505 video có đặc trưng; trên đủ 600
video baseline còn 21.79 (`experiments_report.md:26-28`) [chưa xác minh]. Script chính thức bỏ bước trùng hệt (`eval_article_grounding.py:20-52`);
UniAV không bỏ, nhưng [đếm thực tế] val_seen có 0 bước trùng hệt nên không khác.

## 11. Khác biệt và câu hỏi mở

| # | Khác biệt | Giữa | Ảnh hưởng khi ghép Omni |
|---|---|---|---|
| 1 | `partial`: artc/ov giữ **Full + Partial**; mọi nhánh cũ chỉ **Full Match** | `htstep.py:106` vs `origin/htstep:tools/htstep_hub.py:119` | Có: val_seen 3 441 vs 1 001 bước; 600 vs 405 video. |
| 2 | Text: GT `headline.strip()`, vector `"<activity>: <headline>"`; nhánh cũ: headline trần cho cả hai | `htstep.py:71`, `htstep_data.py:48` vs `origin/htstep:tools/htstep_data.py:52-55` | Có, nếu Omni sinh `omni_emb_file` hoặc so text với UniAV. |
| 3 | Plan `OMNI_0SHOT_HTSTEP_VAL.md` (Fushion/docs) lấy Full Match + headline trần, video lọc trùng YC2, kẹp mốc theo mp4 | `Code/Fushion/docs/OMNI_0SHOT_HTSTEP_VAL.md:80-87` vs `htstep.py:106` | **Lệch** với `htstep-artc` (Partial bị bỏ, text khác, có lọc YC2/kẹp mà UniAV không làm). Khớp với các nhánh cũ. |
| 4 | Plan ghi `anno/htstep.json` nằm trong `htstep-feats-k400`; thực tế nằm trong `nguyenminh04/htstep-feats` | plan dòng 68 vs HF tree (`hf/_info.json`) | Chỉ ảnh hưởng đường dẫn. |
| 5 | `duration` json (100 % số nguyên) có khớp mp4 không | `annotations.json` vs mp4 | [chưa xác minh] không có mp4 HT-Step nào ở local (`Data/Htstep` không có `.mp4`; `D:\Học\KL\videos` không tồn tại). UniAV không dùng `duration` cho lưới, chỉ để kẹp đầu ra. |
| 6 | Lưới thời gian theo `T` = số hàng đặc trưng, không theo `duration` json | `htstep.py:147,153` | Nếu `T ≠ duration` thì giây ↔ ô của UniAV dựa trên `T`; Omni cắt theo giây thì cùng giây gốc. |
| 7 | Tâm cửa sổ video `i + 0.125` s, lưới giả định `i + 0.5` s | `extract_features.py:226` vs `htstep.py:151` | 0.375 s; chỉ ghi nhận. |
| 8 | Dev: nhánh chuẩn **chưa tách dev**; chọn checkpoint trên val_seen | `train.py:52-53,118-122` | Có: không có danh sách dev chung để Omni dùng lại. |
| 9 | Checkpoint / cấu hình HT-Step nhóm sẽ dùng chưa xác định: báo cáo tốt nhất là artc (+512), config mặc định là baseline; báo cáo ghi nhánh `htstep-ov` nhưng artc chỉ có ở `htstep-artc` | `experiments_report.md:3`, `config.py:22,34` | Có: quyết định `max_seq_len`, `article_cond`. |
| 10 | Đặc trưng: UniAV `htstep-artc` dùng `k768` (K400 f16); Fushion HT-Step 0-shot dùng `v768` stage2 | `config.py:20` vs `Fushion configs/htstep_0shot_protocol.json:425-426` | Có: Z / đề xuất của hai bên đến từ hai extractor khác nhau. |
| 11 | Đầu ra `labels` là hàng pool, không phải `global_step_index`; chỉ số bước trong video của UniAV ≠ `#i` của Fushion (`htstep.json` Full Match) | `event_archs.py:562`, `htstep.py:125` vs `uemr_htstep/data.py:75` | Có: cần đổi qua `pool_ids[label]` / map lại chỉ số. |
| 12 | Không có `omni_emb_file` HT-Step; không có file đầu ra inference; không có `z_i` | 9.1, 9.3 | Có: chưa có giao diện file nào giữa UniAV HT-Step và Omni. |
| 13 | `annotations.json` local bị sửa (thêm `headline`, 2026-11-01), không rõ ai sửa | `git status` repo `htstep` | Không với UniAV (không đọc trường đó); có nếu Omni đọc `headline` từ json thay vì taxonomy (giá trị trùng 100 %). |
| 14 | 86 step trùng cả `"<activity>: <headline>"` (khác variation) → vector text trùng hệt | `htstep_data.py:48` | Có nếu dùng vector text để phân biệt step. |
| 15 | Không lọc trùng YouCook2 | `htstep.py:99-132` | 1 video train trùng YC2 (`6MBctYaMU8U`), 0 video val_seen. |
| 16 | Số video không audio | – | [chưa xác minh] cần mở `a768`. |
| 17 | Code không lọc `partial`, nhưng tập train có đặc trưng chỉ gồm video có ≥ 1 Full Match (4 813/4 813), do danh sách trích xuất đến từ `anno/htstep.json` | `htstep.py:106` vs `origin/htstep:tools/htstep_hub.py:117-122`, mục 4.3 | Có: "train của UniAV" thực tế ≠ toàn bộ `subset == train`. val_seen thì có cả video không có Full Match (510 có đặc trưng, 348 có Full Match). |
| 18 | 162 video val_seen có hai bản mp4 (tar `htstep-videos` và `htstep-val-seen-videos`); đặc trưng của chúng lấy từ bản nào: [chưa xác minh] | mục 4.3 | Có nếu hai bản lệch thời gian. |
| 19 | 361 video train có đặc trưng nhưng không thấy mp4 trên HF | mục 4.3 | Có: Omni không đọc được mp4 của các video này từ HF. |
| 20 | `extracted_videos.txt` (4 856) và `htstep-videos/manifest.json` (95 tar) đã cũ so với tar thật (4 992 id; 113 tar) | `hf/`, `feat_tar_index.json` | Có nếu Omni chọn video theo hai file này. |

## 12. Tóm tắt cho Omni: "Omni phải làm giống UniAV"

| Quy tắc | Giá trị chính xác (`htstep-artc`) | Nguồn |
|---|---|---|
| Nguồn nhãn | `data/annotations.json` + `data/taxonomy.csv` gốc `facebookresearch/htstep` (không dùng `anno/htstep.json`) | `config.py:12-13` |
| Split train | `subset == "train"` | `train.py:52`, `htstep.py:104` |
| Split đánh giá | `subset == "val_seen"` | `train.py:53` |
| Dev | không có | `train.py:52-53` |
| Lọc video | có bước sau lọc; có `<id>.npz` đọc được `k768` + `a768`; `min(start) < số hàng đặc trưng`; không lọc YC2 | `htstep.py:103-119` |
| Tập video thực tế | train: 4 813 video có `.npz` (đều có Full Match); val_seen: 510 có `.npz` (báo cáo: 4 810 / 505 sau khi bỏ file hỏng) | mục 4.3 |
| Lọc bước | giữ mọi `partial`; chỉ `segment[1] > segment[0]` | `htstep.py:106` |
| Giới hạn bước | 48 bước đầu theo thứ tự json, sau lọc `end > start`; không sắp, không gộp, không kẹp | `htstep.py:106`, `config.py:25` |
| Đơn vị thời gian | giây float như json; đầu ra kẹp `[0, duration_json]` | `htstep.py:117,203`, `event_archs.py:560` |
| Lưới (nếu cần quy đổi) | `g = t·256/T − 0.5`, `t = (g+0.5)·T/256`, `T = min(len(k768), len(a768))` | `htstep.py:147-162` |
| Text bước (GT / caption) | `taxonomy.headline.strip()` | `htstep.py:71` |
| Text bước (vector) | `"<activity>: <headline.strip()>"`, encoder lower-case, ≤ 40 token | `htstep_data.py:48`, `iv2_text.py:47` |
| Vector bước | trừ mean của 4 958 vector, L2; lưu `ids`, `emb` fp16, `text`, `mean` | `htstep_data.py:53-58` |
| File vector Omni (nếu sinh) | npz `ids` = `global_step_index`, `emb` (N, 3584); UniAV không trừ mean, tự L2 | `htstep.py:17-25`, `event_archs.py:480`, `config.py:46` |
| Ánh xạ id | `annotations[*].id` = `global_step_index`; hàng pool = thứ hạng trong taxonomy sắp tăng (4 958 hàng) | `htstep.py:66-70,120` |
| Article / candidate | `(activity, variation)`, 617 article; `cand` = mọi step của article của video | `htstep.py:75-80,129` |
| Khoá ghép | video id (key json), `global_step_index` (đổi `labels` qua `pool_ids`), mốc giây | mục 9.4 |

---

## Phụ lục A. Script

Mọi script ở `D:\Học\KL\Data\Htstep\khao_sat\`, chạy bằng `C:\Users\Admin\miniconda3\python.exe -I -X utf8 <script>`. Số liệu thô:
`so_lieu_annotations.json`, `so_lieu_dac_trung.json`, `feat_tar_index.json`, `hf/` (file HF nhỏ đã tải), `hf/_info.json`.

### A.1 `tai_hf_metadata.py`

```python
"""Tải các file metadata NHỎ trên HF (README, danh sách, manifest, anno/) vào khao_sat/hf/<repo>/<path>.
Không tải video / đặc trưng. Ghi kèm size + commit cuối của từng file vào hf/_info.json.

    python -I -X utf8 tai_hf_metadata.py
"""
import json
import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

OUT = Path(__file__).resolve().parent / "hf"
FILES = {
    "nguyenminh04/htstep-feats": ["README.md", "anno/htstep.json", "anno/annotations.json"],
    "nguyenminh04/htstep-feats-k400": ["README.md", "extracted_videos.txt"],
    "nguyenminh04/htstep-val-seen-feats": ["README.md"],
    "nguyenminh04/htstep-text-feats": ["README.md"],
    "nguyenminh04/htstep-videos": ["README.md", "manifest.json"],
    "nguyenminh04/htstep-val-seen-videos": ["README.md", "part1/manifest.json", "part2/manifest.json"],
}

api = HfApi()
info = {}
for repo, paths in FILES.items():
    for p in api.get_paths_info(repo, paths, repo_type="dataset", expand=True):
        dst = OUT / repo.split("/")[1] / p.path
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(hf_hub_download(repo, p.path, repo_type="dataset"), dst)
        info[f"{repo}/{p.path}"] = {"size": p.size, "commit": p.last_commit.oid if p.last_commit else None,
                                    "date": str(p.last_commit.date) if p.last_commit else None}
        print(repo, p.path, p.size)
# danh sách file (không tải) của các repo đặc trưng
for repo in ("nguyenminh04/htstep-feats-k400", "nguyenminh04/htstep-val-seen-feats", "nguyenminh04/htstep-text-feats"):
    info[f"{repo}/<tree>"] = sorted((f.path, getattr(f, "size", None))
                                    for f in api.list_repo_tree(repo, repo_type="dataset", recursive=True))
(OUT / "_info.json").write_text(json.dumps(info, indent=1, ensure_ascii=False), encoding="utf-8")
```

### A.2 `doc_tar_header.py`

```python
"""Tên + kích thước các .npz trong mọi tar đặc trưng của htstep-feats-k400 và htstep-val-seen-feats,
bằng cách đọc header tar qua HTTP range (chỉ 512 byte mỗi member; KHÔNG tải nội dung đặc trưng).
Cách đọc lấy từ Data/Htstep/pseudocap/tools/build_tar_index.py.

    python -I -X utf8 doc_tar_header.py   ->  feat_tar_index.json {repo: {tar: [[member, size], ...]}}
"""
import json
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from huggingface_hub import HfApi, get_token, hf_hub_url

OUT = Path(__file__).resolve().parent / "feat_tar_index.json"
REPOS = ("nguyenminh04/htstep-feats-k400", "nguyenminh04/htstep-val-seen-feats")
TOKEN = get_token()


def range_reader(url):
    s = requests.Session()
    r = s.head(url, allow_redirects=False, timeout=60, headers={"Authorization": f"Bearer {TOKEN}"})
    loc = r.headers.get("Location") if r.is_redirect else url

    def read(off, n):
        for attempt in range(5):
            try:
                g = s.get(loc, headers={"Range": f"bytes={off}-{off + n - 1}"}, timeout=60)
                g.raise_for_status()
                assert g.status_code == 206, g.status_code
                return g.content
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
    return read


def members(read):
    out, off, pending = [], 0, None
    while True:
        buf = read(off, 512)
        if len(buf) < 512 or buf == b"\0" * 512:
            return out
        ti = tarfile.TarInfo.frombuf(buf, "utf-8", "surrogateescape")
        data, nxt = off + 512, off + 512 + -(-ti.size // 512) * 512
        if ti.type == tarfile.GNUTYPE_LONGNAME:
            pending = read(data, ti.size).rstrip(b"\0").decode("utf-8")
        elif ti.type == tarfile.XHDTYPE:
            for rec in read(data, ti.size).decode("utf-8").splitlines():
                if " path=" in rec:
                    pending = rec.split(" path=", 1)[1]
        elif ti.isreg():
            out.append([pending or ti.name, ti.size])
            pending = None
        off = nxt


def one(repo, path):
    return repo, path, members(range_reader(hf_hub_url(repo, path, repo_type="dataset")))


api = HfApi()
jobs = [(r, f.path) for r in REPOS for f in api.list_repo_tree(r, repo_type="dataset", path_in_repo="feats")
        if f.path.endswith(".tar")]
res = {r: {} for r in REPOS}
with ThreadPoolExecutor(8) as ex:
    for i, (r, p, m) in enumerate(ex.map(lambda j: one(*j), jobs), 1):
        res[r][p] = m
        print(f"{i}/{len(jobs)} {r} {p}: {len(m)} members", flush=True)
OUT.write_text(json.dumps(res, indent=0), encoding="utf-8")
```

### A.3 `dem_annotations.py`

```python
"""Đếm trên file thật: annotations.json / taxonomy.csv / video_splits.json của HT-Step (bản local và bản gốc ở HEAD
của repo facebookresearch/htstep), theo đúng bộ lọc của UniAV nhánh htstep-artc (libs/datasets/htstep.py:99-132).
Kèm anno/htstep.json trên HF (nhánh cũ htstep / cleanup-htstep / pipeline-k400 đọc file này).

    python -I -X utf8 dem_annotations.py   ->  so_lieu_annotations.json (+ in tóm tắt)
"""
import csv
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent / "htstep"                      # clone facebookresearch/htstep
ANN, TAX, SPL = (REPO / "data" / f for f in ("annotations.json", "taxonomy.csv", "video_splits.json"))
HF_ANNO = HERE / "hf" / "htstep-feats" / "anno" / "htstep.json"
YC2 = HERE.parents[1] / "YouCookII" / "metadata" / "youcookii_annotations_uemr.json"
NUM_CLASSES = 48                                   # libs/core/config.py:25
OUT = {}


def pct(x, qs=(0, 10, 50, 90, 99, 100), seconds=False):
    x = np.asarray(x, dtype=float)
    out = {f"p{q}": round(float(np.percentile(x, q)), 2) for q in qs} | {"n": int(len(x)), "mean": round(float(x.mean()), 2)}
    if seconds:   # tỉ lệ (%) bước dài hơn 8 / 30 / 60 s
        out |= {f">{t}s": round(float((x > t).mean()) * 100, 2) for t in (8, 30, 60)}
    return out


def norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", t.lower())).strip()


# ---------------------------------------------------------------- 0. bản local vs bản gốc (git HEAD)
ann = json.loads(ANN.read_text(encoding="utf-8"))
head = json.loads(subprocess.run(["git", "-C", str(REPO), "show", "HEAD:data/annotations.json"],
                                 capture_output=True, check=True).stdout.decode("utf-8"))
diff_keys = set(ann) ^ set(head)
fields_local = Counter(k for v in ann.values() for a in v["annotations"] for k in a)
fields_head = Counter(k for v in head.values() for a in v["annotations"] for k in a)
same_core = all(
    {k: v[k] for k in v if k != "annotations"} == {k: head[i][k] for k in head[i] if k != "annotations"}
    and [{k: a[k] for k in a if k != "headline"} for a in v["annotations"]] == head[i]["annotations"]
    for i, v in ann.items() if i in head)
OUT["local_vs_head"] = {"videos_local": len(ann), "videos_head": len(head), "keys_differ": len(diff_keys),
                        "step_fields_local": dict(fields_local), "step_fields_head": dict(fields_head),
                        "identical_except_headline_field": same_core}

# ---------------------------------------------------------------- 1. taxonomy -> pool (htstep.py:66-80)
with open(TAX, newline="", encoding="utf-8") as f:
    rows = sorted(csv.DictReader(f), key=lambda r: int(r["global_step_index"]))
pool_ids = [int(r["global_step_index"]) for r in rows]
step_row = {s: i for i, s in enumerate(pool_ids)}
art_of = {int(r["global_step_index"]): (r["activity"], r["variation"]) for r in rows}
articles = Counter(art_of.values())
heads = [r["headline"].strip() for r in rows]
act_heads = ["%s: %s" % (r["activity"], r["headline"].strip()) for r in rows]
hc = Counter(heads)
OUT["taxonomy"] = {
    "steps": len(rows), "id_min": min(pool_ids), "id_max": max(pool_ids),
    "id_gaps": max(pool_ids) + 1 - len(pool_ids), "ids_unique": len(set(pool_ids)) == len(pool_ids),
    "activities": len({r["activity"] for r in rows}), "articles": len(articles),
    "steps_per_article": pct(list(articles.values()), (0, 50, 100)),
    "variation_values": dict(Counter(r["variation"] for r in rows).most_common(8)),
    "distinct_headline_strip": len(hc), "distinct_activity_headline": len(set(act_heads)),
    "distinct_headline_lower_nopunct": len({norm(h) for h in heads}),
    "headline_raw_ne_strip": sum(r["headline"] != r["headline"].strip() for r in rows),
    "steps_whose_headline_is_shared": sum(c for c in hc.values() if c > 1),
    "top20_headline": hc.most_common(20)}

# ---------------------------------------------------------------- 2. subset / video_splits
spl = json.loads(SPL.read_text(encoding="utf-8"))
OUT["subsets"] = {"annotations.json": dict(Counter(v["subset"] for v in ann.values())),
                  "video_splits.json": {k: len(v) for k, v in spl.items()},
                  "video_splits_vs_subset": {k: dict(Counter(ann[i]["subset"] if i in ann else "<not in annotations>"
                                                             for i in v)) for k, v in spl.items()}}
if YC2.exists():
    yc2 = set(json.loads(YC2.read_text(encoding="utf-8"))["database"])
    OUT["yc2_overlap"] = {"yc2_ids": len(yc2), "file": str(YC2.name),
                          **{s: sorted(i for i, v in ann.items() if v["subset"] == s and i in yc2)
                             for s in ("train", "val_seen")}}

# ---------------------------------------------------------------- 3. bộ lọc bước theo UniAV, từng split
partial_vals = Counter(a["partial"] for v in ann.values() for a in v["annotations"])
OUT["partial_values"] = dict(partial_vals)
for split in ("train", "val_seen"):
    vids = {i: v for i, v in ann.items() if v["subset"].lower() == split}
    s = defaultdict(int)
    lens_all, lens_kept, lens_full, n_steps_kept, cand, multi_art, act_mismatch = [], [], [], [], [], [], 0
    head_mismatch, unsorted, overlap_pairs, overlap_videos, beyond_dur, start_ge_dur, neg_start = 0, 0, 0, 0, 0, 0, 0
    kept_videos = 0
    for i, v in vids.items():
        A = v["annotations"]
        s["steps_raw"] += len(A)
        s["full_raw"] += sum(a["partial"] == "Full Match" for a in A)
        lens_all += [a["segment"][1] - a["segment"][0] for a in A]
        ok = [a for a in A if a["segment"][1] > a["segment"][0]]      # htstep.py:106
        s["drop_end_le_start"] += len(A) - len(ok)
        s["videos_over_48"] += len(ok) > NUM_CLASSES
        s["drop_over_48"] += max(0, len(ok) - NUM_CLASSES)
        ok = ok[:NUM_CLASSES]
        if not ok:
            s["videos_no_step"] += 1
            continue
        kept_videos += 1
        n_steps_kept.append(len(ok))
        st = [a["segment"][0] for a in ok]
        unsorted += st != sorted(st)
        segs = sorted((a["segment"] for a in ok))
        ov = sum(b[0] < a[1] for a, b in zip(segs, segs[1:]))
        overlap_pairs += ov
        overlap_videos += ov > 0
        for a in ok:
            ts, te = a["segment"]
            lens_kept.append(te - ts)
            if a["partial"] == "Full Match":
                lens_full.append(te - ts)
            beyond_dur += te > v["duration"]
            start_ge_dur += ts >= v["duration"]
            neg_start += ts < 0
            if a.get("headline") is not None and a["headline"] != rows[step_row[a["id"]]]["headline"]:
                head_mismatch += 1
            act_mismatch += art_of[a["id"]] != (v["activity"], v["variation"])
        arts = {art_of[a["id"]] for a in ok}
        multi_art.append(len(arts) > 1)
        cand.append(articles[art_of[ok[0]["id"]]])
        s["steps_kept"] += len(ok)
        s["full_kept"] += sum(a["partial"] == "Full Match" for a in ok)
    ids_missing = sum(a["id"] not in step_row for v in vids.values() for a in v["annotations"])
    # bước trùng hệt (cùng id, hai mốc lệch <= 1e-3 s): eval_article_grounding.py:20-40 bỏ, UniAV không bỏ
    dup = sum(1 for v in vids.values() for j, a in enumerate(v["annotations"]) for b in v["annotations"][:j]
              if a["id"] == b["id"] and abs(a["segment"][0] - b["segment"][0]) <= 1e-3
              and abs(a["segment"][1] - b["segment"][1]) <= 1e-3)
    OUT[split] = {
        "videos_in_json": len(vids), "videos_after_step_filter": kept_videos, **dict(s),
        "partial": dict(Counter(a["partial"] for v in vids.values() for a in v["annotations"])),
        "videos_with_full_match": sum(any(a["partial"] == "Full Match" for a in v["annotations"]) for v in vids.values()),
        "step_ids_not_in_taxonomy": ids_missing, "exact_duplicate_steps": dup,
        "distinct_step_ids_used": len({a["id"] for v in vids.values() for a in v["annotations"]}),
        "steps_per_video_kept": pct(n_steps_kept, (0, 50, 90, 100)),
        "videos_steps_not_sorted_by_start": unsorted,
        "overlap_pairs_sorted": overlap_pairs, "videos_with_overlap": overlap_videos,
        "steps_end_gt_duration": beyond_dur, "steps_start_ge_duration": start_ge_dur, "steps_start_lt_0": neg_start,
        "headline_field_ne_taxonomy": head_mismatch, "step_article_ne_video_activity_variation": act_mismatch,
        "videos_steps_from_several_articles": int(sum(multi_art)),
        "cand_size": pct(cand, (0, 50, 100)),
        "len_raw": pct(lens_all, seconds=True), "len_kept": pct(lens_kept, seconds=True),
        "len_kept_full_match": pct(lens_full, seconds=True),
        "duration_json": pct([v["duration"] for v in vids.values()], (0, 50, 100)),
        "duration_is_integer": sum(float(v["duration"]).is_integer() for v in vids.values())}

# trùng headline ở mức bước đã gán nhãn (val_seen, sau lọc UniAV): bước có headline trùng một bước của video khác
for split in ("train", "val_seen"):
    by_head = defaultdict(set)
    rows_ = []
    for i, v in ann.items():
        if v["subset"] != split:
            continue
        for a in [a for a in v["annotations"] if a["segment"][1] > a["segment"][0]][:NUM_CLASSES]:
            h = rows[step_row[a["id"]]]["headline"].strip()
            by_head[h].add(i)
            rows_.append((i, h, a["id"]))
    OUT[split]["steps_sharing_headline_with_other_video_pct"] = round(
        100 * sum(len(by_head[h] - {i}) > 0 for i, h, _ in rows_) / len(rows_), 2)
    by_id = defaultdict(set)
    for i, _, s_ in rows_:
        by_id[s_].add(i)
    OUT[split]["steps_sharing_step_id_with_other_video_pct"] = round(
        100 * sum(len(by_id[s_] - {i}) > 0 for i, _, s_ in rows_) / len(rows_), 2)

# ---------------------------------------------------------------- 4. anno/htstep.json (HF, nhánh cũ)
if HF_ANNO.exists():
    hf = json.loads(HF_ANNO.read_text(encoding="utf-8"))
    full = {i: [a for a in ann[i]["annotations"] if a["partial"] == "Full Match"] for i in hf if i in ann}
    same = sum(len(full[i]) == len(hf[i]["segments"]) and all(
        abs(a["segment"][0] - s[0]) < 1e-6 and abs(a["segment"][1] - s[1]) < 1e-6
        and rows[step_row[a["id"]]]["headline"] == c
        for a, s, c in zip(full[i], hf[i]["segments"], hf[i]["captions"])) for i in full)
    OUT["hf_anno_htstep_json"] = {
        "videos": len(hf), "roles": dict(Counter(v["role"] for v in hf.values())),
        "steps": sum(len(v["segments"]) for v in hf.values()),
        "role_eq_subset": sum(hf[i]["role"] == ann[i]["subset"] for i in hf if i in ann),
        "not_in_annotations": sum(i not in ann for i in hf),
        "videos_equal_to_full_match_headline_of_official": same}

(HERE / "so_lieu_annotations.json").write_text(json.dumps(OUT, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(OUT, indent=1, ensure_ascii=False)[:20000])
```

### A.4 `dem_dac_trung.py`

```python
"""Video HT-Step nào có đặc trưng k400 trên HF (theo tên file trong tar, không mở đặc trưng) và có mp4 ở đâu.
Mô phỏng tools/htstep_data.py fetch (htstep-artc:69-93): mọi tar của htstep-feats-k400 rồi htstep-val-seen-feats,
theo thứ tự tên, chỉ .npz size > 0 và có trong annotations.json; file sau ghi đè file trước.

    python -I -X utf8 dem_dac_trung.py   (cần feat_tar_index.json của doc_tar_header.py và hf/ của tai_hf_metadata.py)
    -> so_lieu_dac_trung.json
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
HF = HERE / "hf"
ann = json.loads((HERE.parent / "htstep" / "data" / "annotations.json").read_text(encoding="utf-8"))
idx = json.loads((HERE / "feat_tar_index.json").read_text(encoding="utf-8"))
extracted = set((HF / "htstep-feats-k400" / "extracted_videos.txt").read_text(encoding="utf-8").split())
OUT = {}

final_src = {}                                   # video -> (repo, tar) của file cuối cùng sau fetch
nonempty, empty, seen_in = defaultdict(set), defaultdict(set), defaultdict(list)
for repo in ("nguyenminh04/htstep-feats-k400", "nguyenminh04/htstep-val-seen-feats"):   # FEATS_REPOS
    for tar in sorted(idx[repo]):
        for name, size in idx[repo][tar]:
            if not name.endswith(".npz"):
                continue
            v = Path(name).name[:-4]
            seen_in[v].append(tar)
            (nonempty if size > 0 else empty)[repo].add(v)
            if size > 0 and v in ann:
                final_src[v] = (repo, tar)
OUT["tars"] = {r: {"tars": len(t), "npz_members": sum(len(m) for m in t.values()),
                   "non_empty_ids": len(nonempty[r]), "empty_ids": len(empty[r] - nonempty[r])}
               for r, t in idx.items()}
OUT["extracted_videos.txt"] = {"ids": len(extracted),
                               "eq_k400_non_empty": extracted == nonempty["nguyenminh04/htstep-feats-k400"],
                               "only_in_txt": len(extracted - nonempty["nguyenminh04/htstep-feats-k400"]),
                               "only_in_tars": len(nonempty["nguyenminh04/htstep-feats-k400"] - extracted)}
OUT["ids_in_several_tars"] = sum(len(set(t)) > 1 for t in seen_in.values())

# nguồn mp4
man = json.loads((HF / "htstep-videos" / "manifest.json").read_text(encoding="utf-8"))["shards"]
in_manifest = {v for s in man.values() for v in s["videos"]}   # manifest.json cũ: 95 trong 113 tar
# chỉ mục header của đủ 113 tar video (Data/Htstep/pseudocap/tools/build_tar_index.py)
TAR_INDEX = HERE.parent / "pseudocap" / "metadata" / "tar_index.csv"
import csv
with open(TAR_INDEX, newline="", encoding="utf-8") as f:
    tar_rows = list(csv.DictReader(f))
in_shards = {r["video_id"] for r in tar_rows if r["member_name"].endswith(".mp4")}
OUT["htstep_videos"] = {"manifest_shards": len(man), "manifest_ids": len(in_manifest),
                        "tar_index_tars": len({r["tar_path"] for r in tar_rows}), "tar_index_mp4_ids": len(in_shards),
                        "manifest_minus_tar_index": len(in_manifest - in_shards)}
p1 = json.loads((HF / "htstep-val-seen-videos" / "part1" / "manifest.json").read_text(encoding="utf-8"))
p2 = json.loads((HF / "htstep-val-seen-videos" / "part2" / "manifest.json").read_text(encoding="utf-8"))

for split in ("train", "val_seen"):
    vids = {i for i, v in ann.items() if v["subset"] == split}
    have = {v for v in vids if v in final_src}
    by = Counter(final_src[v][1] if final_src[v][0].endswith("val-seen-feats") else "k400:" + final_src[v][0].split("-")[-1]
                 for v in have)
    OUT[split] = {
        "videos": len(vids),
        "with_nonempty_npz_any_repo (upper bound of UniAV use)": len(have),
        "in_k400_repo": len(vids & nonempty["nguyenminh04/htstep-feats-k400"]),
        "in_val_seen_repo": len(vids & nonempty["nguyenminh04/htstep-val-seen-feats"]),
        "final_file_from": dict(by.most_common()) if split == "val_seen" else {
            "k400": sum(final_src[v][0].endswith("k400") for v in have),
            "val-seen-feats": sum(final_src[v][0].endswith("val-seen-feats") for v in have)},
        "only_empty_npz": len({v for v in vids if v in empty["nguyenminh04/htstep-feats-k400"] | empty[
            "nguyenminh04/htstep-val-seen-feats"]} - have),
        "with_full_match_and_npz": sum(any(a["partial"] == "Full Match" for a in ann[v]["annotations"]) for v in have),
        "mp4_in_htstep_videos_shards": len(vids & in_shards),
        "mp4_in_val_seen_videos_part1_ok": len(vids & set(p1["ok"])),
        "mp4_in_val_seen_videos_part2_ok": len(vids & set(p2["ok"])),
        "npz_but_no_known_mp4": len(have - in_shards - set(p1["ok"]) - set(p2["ok"])),
        "mp4_any_source": len(vids & (in_shards | set(p1["ok"]) | set(p2["ok"]))),
        "mp4_in_shards_and_val_seen_repo": len(vids & in_shards & (set(p1["ok"]) | set(p2["ok"]))),
        "npz_from_k400_but_mp4_not_in_shards": len({v for v in have if final_src[v][0].endswith("k400")} - in_shards),
    }
from huggingface_hub import HfApi   # chỉ liệt kê tên file, không tải

vfiles = HfApi().list_repo_files("nguyenminh04/htstep-val-seen-videos", repo_type="dataset")
OUT["val_seen_videos_repo"] = {
    "part1_ok": len(p1["ok"]), "part1_failed": len(p1.get("failed", {})), "part2_ok": len(p2["ok"]),
    "files_by_folder_ext": dict(Counter(f"{Path(f).parent}/*{''.join(Path(f).suffixes)}" for f in vfiles)),
    "m4a_files": [f for f in vfiles if f.endswith(".m4a")],
    "ok_without_mp4": sorted(v for d, m in (("part1", p1), ("part2", p2)) for v in m["ok"]
                             if f"{d}/{v}.mp4" not in vfiles),
}
(HERE / "so_lieu_dac_trung.json").write_text(json.dumps(OUT, indent=1, ensure_ascii=False), encoding="utf-8")
print(json.dumps(OUT, indent=1, ensure_ascii=False))
```


## Phụ lục B. Lệnh và output

### B.1 Lệnh git (UniAV-fixed, Fushion, repo htstep gốc)

```text
$ cd Code/UniAV/UniAV-fixed && git branch -vv && git status --short && git log -5 --oneline
  chien           cd11ee8 [origin/chien] UEMR 2 cross-fitting: hai fold Athena yc2clean (A, B: 501 + 501 video trainsplit)
  chien-remote-cu ecb7433 [origin/chien: ahead 3, behind 94] add mAP checkpoint
* htstep          8d688ff HT-Step article conditioning (artc), article head, mAP checkpoint selection
  main            ecb7433 [origin/chien: ahead 3, behind 94] add mAP checkpoint
8d688ff HT-Step article conditioning (artc), article head, mAP checkpoint selection
99a8390 Step text vectors from htstep-text-feats; drop dim notes at the top of event_archs.py
742f302 han li
cd5ebda HT-Step minimal: minimal layout on HT-Step k768 features, DDP via torchrun, Python 3.13 / torch 2.14
3f8cd35 fix

$ for b in <nhánh HT-Step>; do git log -1 --format="%h %ci %s" $b; done
origin/htstep: 13114d8 2026-10-21 09:44:13 +0700 HT-Step only: remove YouCook2, COIN, ActivityNet and benchmark code; split t
origin/pipeline-k400: bf129b1 2026-10-21 13:31:50 +0700 Pipeline: upload k400 features to htstep-feats-k400, A100 launcher, max-ahea
origin/cleanup-htstep: 7c415ac 2026-10-29 08:30:02 +0700 Pipeline: --upload-every N (partial tar to HF every N videos), upload_partia
origin/htstep-minimal: cd5ebda 2026-10-30 08:05:23 +0700 HT-Step minimal: minimal layout on HT-Step k768 features, DDP via torchrun, 
origin/htstep-ov: 742f302 2026-10-30 21:19:54 +0700 han li
origin/htstep-artc: 8d688ff 2026-11-01 12:53:30 +0700 HT-Step article conditioning (artc), article head, mAP checkpoint selection
origin/chien: cd11ee8 2026-10-23 12:40:27 +0700 UEMR 2 cross-fitting: hai fold Athena yc2clean (A, B: 501 + 501 video trains
origin/main: 2ad4432 2026-10-03 13:44:34 +0700 add comment

$ git merge-base origin/htstep origin/cleanup-htstep; git merge-base origin/htstep origin/pipeline-k400
13114d8ef715ec5b40a5025199b4d9e71bfede4d
13114d8ef715ec5b40a5025199b4d9e71bfede4d

$ for b in origin/htstep-artc origin/htstep-ov origin/cleanup-htstep; do git ls-tree -r --name-only $b | grep -c -i "captioner\|caption_pool"; done
origin/htstep-artc: 0
origin/htstep-ov: 0
origin/cleanup-htstep: 0

$ git grep -n -i "omni" origin/htstep -- "libs/*.py" "tools/*.py" "train*.py" "configs/*"   (chỉ đọc file, không sinh)
origin/htstep:tools/export_api_ckpt.py:5:Keeps the best captioning checkpoint (best_cap) in fp16, without the OmniRetriever projection
origin/htstep:tools/export_api_ckpt.py:10:the teacher's text space (caption_space omni) also gets those caption vectors (caption_pool, in
origin/htstep:tools/export_api_ckpt.py:30:    """Teacher-space caption vectors in the order of the API caption pool (caption_space omni), else None.
origin/htstep:tools/export_api_ckpt.py:50:    z = np.load(os.path.join(ROOT, cfg['dataset']['omni_emb_file']))
origin/htstep:tools/export_api_ckpt.py:68:                      if not k.startswith('omni_')}
origin/htstep:tools/summarize_runs.py:35:                 teacher='yes' if (d.get('config') or {}).get('dataset', {}).get('omni_emb_file') else 'no')

$ cd Code/Fushion && git grep -n -i "model_best|htstep-artc|k768|label_probs|step_emb_iv2j|global_step_index" htstep -- . ":!*.ipynb"
htstep:configs/htstep_0shot_protocol.json:426:   "not_used": "feats_k400/ và repo htstep-feats-k400 (InternVideo2-Stage1-1B K400 f16, khoá k768): extractor kh
htstep:uemr_htstep/data.py:121:    ``global_step_index -> headline`` of ``taxonomy.csv``. Returns ``(sampled, mismatches)``."""

$ cd Data/Htstep/htstep && git status --short && git log -1 --format="%h %ci %s"
 M data/annotations.json
cca1dea 2024-02-27 04:29:33 +0000 Init repo
$ git show HEAD:data/taxonomy.csv | tr -d "\r" | sha256sum; tr -d "\r" < data/taxonomy.csv | sha256sum
f19e254020848a1bdb287810586eba3d2592706fc58071cb82199ba1b767ee66 *-
f19e254020848a1bdb287810586eba3d2592706fc58071cb82199ba1b767ee66 *-

$ find Data/Htstep -maxdepth 5 -name "*.mp4" -o -name "*.npz" -o -name "*.tar"   -> (không có kết quả); ls D:/Học/KL/videos -> không tồn tại
```

### B.2 `tai_hf_metadata.py` (file đã tải, byte)

```text
nguyenminh04/htstep-feats/README.md 2428
nguyenminh04/htstep-feats/anno/htstep.json 3894832
nguyenminh04/htstep-feats/anno/annotations.json 350905
nguyenminh04/htstep-feats-k400/README.md 4228
nguyenminh04/htstep-feats-k400/extracted_videos.txt 58272
nguyenminh04/htstep-val-seen-feats/README.md 1528
nguyenminh04/htstep-text-feats/README.md 2636
nguyenminh04/htstep-videos/README.md 2401
nguyenminh04/htstep-videos/manifest.json 94935
nguyenminh04/htstep-val-seen-videos/README.md 808
nguyenminh04/htstep-val-seen-videos/part1/manifest.json 9083
nguyenminh04/htstep-val-seen-videos/part2/manifest.json 2422
```

### B.3 `doc_tar_header.py` (log)

```text
1/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_1000.tar: 91 members
2/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_1002.tar: 87 members
3/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_1004.tar: 76 members
4/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_1006.tar: 72 members
5/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_1008.tar: 70 members
6/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_1010.tar: 3 members
7/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_2001.tar: 32 members
8/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_3001.tar: 87 members
9/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_3003.tar: 86 members
10/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_3005.tar: 59 members
11/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_3006.tar: 93 members
12/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_3009.tar: 87 members
13/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_3013.tar: 88 members
14/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_3014.tar: 5 members
15/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_3018.tar: 2 members
16/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_3021.tar: 87 members
17/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_3022.tar: 87 members
18/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_3025.tar: 3 members
19/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_3029.tar: 84 members
20/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_3030.tar: 5 members
21/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_3036.tar: 92 members
22/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_3041.tar: 85 members
23/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_3042.tar: 82 members
24/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_3046.tar: 93 members
25/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_3049.tar: 18 members
26/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_3050.tar: 94 members
27/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_3057.tar: 3 members
28/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_3059.tar: 3 members
29/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_3061.tar: 2 members
30/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_3065.tar: 5 members
31/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_3067.tar: 32 members
32/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_3068.tar: 86 members
33/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_3070.tar: 80 members
34/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_3078.tar: 86 members
35/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_3082.tar: 93 members
36/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_3084.tar: 2 members
37/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_3088.tar: 85 members
38/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_3096.tar: 87 members
39/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_3105.tar: 79 members
40/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_3107.tar: 72 members
41/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_3109.tar: 83 members
42/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_4000.tar: 88 members
43/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_4006.tar: 84 members
44/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_4008.tar: 86 members
45/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_4010.tar: 87 members
46/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_4012.tar: 93 members
47/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_4014.tar: 90 members
48/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_4016.tar: 13 members
49/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_4018.tar: 39 members
50/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_4019.tar: 86 members
51/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_4021.tar: 86 members
52/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_4023.tar: 84 members
53/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_4024.tar: 86 members
54/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_4025.tar: 78 members
55/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_4027.tar: 86 members
56/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_4028.tar: 35 members
57/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_4029.tar: 84 members
58/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_4030.tar: 90 members
59/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_4031.tar: 84 members
60/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_4032.tar: 83 members
61/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_4033.tar: 68 members
62/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_4034.tar: 90 members
63/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_4035.tar: 86 members
64/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_4036.tar: 42 members
65/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_4037.tar: 85 members
66/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_4038.tar: 44 members
67/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_4039.tar: 86 members
68/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_4040.tar: 22 members
69/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_4041.tar: 88 members
70/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_4042.tar: 86 members
71/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_4043.tar: 91 members
72/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_4044.tar: 83 members
73/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_4051.tar: 9 members
74/80 nguyenminh04/htstep-feats-k400 feats/htstep_feats_iv2_train.tar: 705 members
75/80 nguyenminh04/htstep-val-seen-feats feats/part1_colab.tar: 67 members
76/80 nguyenminh04/htstep-val-seen-feats feats/part1_pc.tar: 104 members
77/80 nguyenminh04/htstep-val-seen-feats feats/part2_colab.tar: 10 members
78/80 nguyenminh04/htstep-val-seen-feats feats/part2_pc.tar: 18 members
79/80 nguyenminh04/htstep-val-seen-feats feats/part2_vast.tar: 101 members
80/80 nguyenminh04/htstep-val-seen-feats feats/val_iv2.tar: 78 members
```

### B.4 `dem_annotations.py`

```text
{
 "local_vs_head": {
  "videos_local": 18072,
  "videos_head": 18072,
  "keys_differ": 0,
  "step_fields_local": {
   "segment": 106734,
   "partial": 106734,
   "id": 106734,
   "headline": 106734
  },
  "step_fields_head": {
   "segment": 106734,
   "partial": 106734,
   "id": 106734
  },
  "identical_except_headline_field": true
 },
 "taxonomy": {
  "steps": 4958,
  "id_min": 0,
  "id_max": 5784,
  "id_gaps": 827,
  "ids_unique": true,
  "activities": 433,
  "articles": 617,
  "steps_per_article": {
   "p0": 1.0,
   "p50": 7.0,
   "p100": 27.0,
   "n": 617,
   "mean": 8.04
  },
  "variation_values": {
   "none": 3315,
   "A": 641,
   "B": 555,
   "C": 310,
   "D": 108,
   "E": 23,
   "F": 6
  },
  "distinct_headline_strip": 4594,
  "distinct_activity_headline": 4872,
  "distinct_headline_lower_nopunct": 4582,
  "headline_raw_ne_strip": 0,
  "steps_whose_headline_is_shared": 501,
  "top20_headline": [
   [
    "Finished.",
    58
   ],
   [
    "Serve.",
    56
   ],
   [
    "Serve warm.",
    19
   ],
   [
    "Heat the oil.",
    18
   ],
   [
    "Serve hot.",
    12
   ],
   [
    "Serve immediately.",
    7
   ],
   [
    "Knead the dough.",
    7
   ],
   [
    "Roll out the dough.",
    6
   ],
   [
    "Boil the water.",
    6
   ],
   [
    "Gather your ingredients.",
    6
   ],
   [
    "Combine the dry ingredients.",
    6
   ],
   [
    "Preheat your oven.",
    5
   ],
   [
    "Chop the vegetables.",
    5
   ],
   [
    "Remove from the heat.",
    5
   ],
   [
    "Heat the oil in a skillet.",
    5
   ],
   [
    "Combine the ingredients.",
    5
   ],
   [
    "Roll the dough into balls.",
    4
   ],
   [
    "Add the remaining ingredients.",
    4
   ],
   [
    "Serve and enjoy.",
    4
   ],
   [
    "Mix well.",
    4
   ]
  ]
 },
 "subsets": {
  "annotations.json": {
   "train": 17472,
   "val_seen": 600
  },
  "video_splits.json": {
   "test_seen": 600,
   "test_unseen": 1000,
   "train": 17472,
   "val_seen": 600,
   "val_unseen": 963
  },
  "video_splits_vs_subset": {
   "test_seen": {
    "<not in annotations>": 600
   },
   "test_unseen": {
    "<not in annotations>": 1000
   },
   "train": {
    "train": 17472
   },
   "val_seen": {
    "val_seen": 600
   },
   "val_unseen": {
    "<not in annotations>": 963
   }
  }
 },
 "yc2_overlap": {
  "yc2_ids": 1790,
  "file": "youcookii_annotations_uemr.json",
  "train": [
   "6MBctYaMU8U"
  ],
  "val_seen": []
 },
 "partial_values": {
  "Partial Match": 77964,
  "Full Match": 28770
 },
 "train": {
  "videos_in_json": 17472,
  "videos_after_step_filter": 17472,
  "steps_raw": 103293,
  "full_raw": 27769,
  "drop_end_le_start": 0,
  "videos_over_48": 1,
  "drop_over_48": 35,
  "steps_kept": 103258,
  "full_kept": 27769,
  "partial": {
   "Partial Match": 75524,
   "Full Match": 27769
  },
  "videos_with_full_match": 11618,
  "step_ids_not_in_taxonomy": 0,
  "exact_duplicate_steps": 0,
  "distinct_step_ids_used": 4531,
  "steps_per_video_kept": {
   "p0": 1.0,
   "p50": 5.0,
   "p90": 11.0,
   "p100": 48.0,
   "n": 17472,
   "mean": 5.91
  },
  "videos_steps_not_sorted_by_start": 3880,
  "overlap_pairs_sorted": 1325,
  "videos_with_overlap": 1052,
  "steps_end_gt_duration": 76,
  "steps_start_ge_duration": 0,
  "steps_start_lt_0": 0,
  "headline_field_ne_taxonomy": 0,
  "step_article_ne_video_activity_variation": 0,
  "videos_steps_from_several_articles": 0,
  "cand_size": {
   "p0": 1.0,
   "p50": 9.0,
   "p100": 27.0,
   "n": 17472,
   "mean": 9.58
  },
  "len_raw": {
   "p0": 0.0,
   "p10": 1.81,
   "p50": 7.47,
   "p90": 34.82,
   "p99": 110.09,
   "p100": 800.33,
   "n": 103293,
   "mean": 14.88,
   ">8s": 47.68,
   ">30s": 12.42,
   ">60s": 3.86
  },
  "len_kept": {
   "p0": 0.0,
   "p10": 1.81,
   "p50": 7.47,
   "p90": 34.82,
   "p99": 110.09,
   "p100": 800.33,
   "n": 103258,
   "mean": 14.88,
   ">8s": 47.69,
   ">30s": 12.42,
   ">60s": 3.86
  },
  "len_kept_full_match": {
   "p0": 0.0,
   "p10": 2.6,
   "p50": 10.67,
   "p90": 45.53,
   "p99": 135.75,
   "p100": 800.33,
   "n": 27769,
   "mean": 19.66,
   ">8s": 59.77,
   ">30s": 18.27,
   ">60s": 6.11
  },
  "duration_json": {
   "p0": 28.0,
   "p50": 342.0,
   "p100": 1957.0,
   "n": 17472,
   "mean": 393.89
  },
  "duration_is_integer": 17472,
  "steps_sharing_headline_with_other_video_pct": 99.44,
  "steps_sharing_step_id_with_other_video_pct": 99.36
 },
 "val_seen": {
  "videos_in_json": 600,
  "videos_after_step_filter": 600,
  "steps_raw": 3441,
  "full_raw": 1001,
  "drop_end_le_start": 0,
  "videos_over_48": 0,
  "drop_over_48": 0,
  "steps_kept": 3441,
  "full_kept": 1001,
  "partial": {
   "Partial Match": 2440,
   "Full Match": 1001
  },
  "videos_with_full_match": 405,
  "step_ids_not_in_taxonomy": 0,
  "exact_duplicate_steps": 0,
  "distinct_step_ids_used": 983,
  "steps_per_video_kept": {
   "p0": 1.0,
   "p50": 5.0,
   "p90": 11.0,
   "p100": 27.0,
   "n": 600,
   "mean": 5.74
  },
  "videos_steps_not_sorted_by_start": 144,
  "overlap_pairs_sorted": 42,
  "videos_with_overlap": 37,
  "steps_end_gt_duration": 3,
  "steps_start_ge_duration": 0,
  "steps_start_lt_0": 0,
  "headline_field_ne_taxonomy": 0,
  "step_article_ne_video_activity_variation": 0,
  "videos_steps_from_several_articles": 0,
  "cand_size": {
   "p0": 2.0,
   "p50": 9.0,
   "p100": 27.0,
   "n": 600,
   "mean": 10.13
  },
  "len_raw": {
   "p0": 0.01,
   "p10": 1.91,
   "p50": 7.69,
   "p90": 36.49,
   "p99": 119.73,
   "p100": 469.78,
   "n": 3441,
   "mean": 15.9,
   ">8s": 48.68,
   ">30s": 12.79,
   ">60s": 4.36
  },
  "len_kept": {
   "p0": 0.01,
   "p10": 1.91,
   "p50": 7.69,
   "p90": 36.49,
   "p99": 119.73,
   "p100": 469.78,
   "n": 3441,
   "mean": 15.9,
   ">8s": 48.68,
   ">30s": 12.79,
   ">60s": 4.36
  },
  "len_kept_full_match": {
   "p0": 0.17,
   "p10": 2.74,
   "p50": 10.76,
   "p90": 45.14,
   "p99": 147.44,
   "p100": 398.59,
   "n": 1001,
   "mean": 20.43,
   ">8s": 60.94,
   ">30s": 18.18,
   ">60s": 5.79
  },
  "duration_json": {
   "p0": 37.0,
   "p50": 331.0,
   "p100": 1759.0,
   "n": 600,
   "mean": 382.75
  },
  "duration_is_integer": 600,
  "steps_sharing_headline_with_other_video_pct": 86.4,
  "steps_sharing_step_id_with_other_video_pct": 85.73
 },
 "hf_anno_htstep_json": {
  "videos": 12023,
  "roles": {
   "val_seen": 405,
   "train": 11618
  },
  "steps": 28770,
  "role_eq_subset": 12023,
  "not_in_annotations": 0,
  "videos_equal_to_full_match_headline_of_official": 12023
 }
}
```

### B.5 `dem_dac_trung.py`

```text
{
 "tars": {
  "nguyenminh04/htstep-feats-k400": {
   "tars": 74,
   "npz_members": 5513,
   "non_empty_ids": 4992,
   "empty_ids": 0
  },
  "nguyenminh04/htstep-val-seen-feats": {
   "tars": 6,
   "npz_members": 378,
   "non_empty_ids": 331,
   "empty_ids": 0
  }
 },
 "extracted_videos.txt": {
  "ids": 4856,
  "eq_k400_non_empty": false,
  "only_in_txt": 0,
  "only_in_tars": 136
 },
 "ids_in_several_tars": 484,
 "htstep_videos": {
  "manifest_shards": 95,
  "manifest_ids": 5907,
  "tar_index_tars": 113,
  "tar_index_mp4_ids": 7453,
  "manifest_minus_tar_index": 0
 },
 "train": {
  "videos": 17472,
  "with_nonempty_npz_any_repo (upper bound of UniAV use)": 4813,
  "in_k400_repo": 4813,
  "in_val_seen_repo": 0,
  "final_file_from": {
   "k400": 4813,
   "val-seen-feats": 0
  },
  "only_empty_npz": 0,
  "with_full_match_and_npz": 4813,
  "mp4_in_htstep_videos_shards": 7110,
  "mp4_in_val_seen_videos_part1_ok": 0,
  "mp4_in_val_seen_videos_part2_ok": 0,
  "npz_but_no_known_mp4": 361,
  "mp4_any_source": 7110,
  "mp4_in_shards_and_val_seen_repo": 0,
  "npz_from_k400_but_mp4_not_in_shards": 361
 },
 "val_seen": {
  "videos": 600,
  "with_nonempty_npz_any_repo (upper bound of UniAV use)": 510,
  "in_k400_repo": 179,
  "in_val_seen_repo": 331,
  "final_file_from": {
   "k400:k400": 179,
   "feats/part1_pc.tar": 103,
   "feats/val_iv2.tar": 78,
   "feats/part1_colab.tar": 66,
   "feats/part2_vast.tar": 58,
   "feats/part2_pc.tar": 17,
   "feats/part2_colab.tar": 9
  },
  "only_empty_npz": 0,
  "with_full_match_and_npz": 348,
  "mp4_in_htstep_videos_shards": 343,
  "mp4_in_val_seen_videos_part1_ok": 169,
  "mp4_in_val_seen_videos_part2_ok": 160,
  "npz_but_no_known_mp4": 0,
  "mp4_any_source": 510,
  "mp4_in_shards_and_val_seen_repo": 162,
  "npz_from_k400_but_mp4_not_in_shards": 0
 },
 "val_seen_videos_repo": {
  "part1_ok": 169,
  "part1_failed": 42,
  "part2_ok": 160,
  "files_by_folder_ext": {
   "./*": 1,
   "./*.md": 1,
   "part1/*.mp4": 169,
   "part1/*.f140.m4a": 2,
   "part1/*.f605.mp4": 1,
   "part1/*.json": 1,
   "part2/*.mp4": 160,
   "part2/*.json": 1
  },
  "m4a_files": [
   "part1/G_6gAP7wMAs.f140.m4a",
   "part1/R--fZfXV_jg.f140.m4a"
  ],
  "ok_without_mp4": []
 }
}
```

