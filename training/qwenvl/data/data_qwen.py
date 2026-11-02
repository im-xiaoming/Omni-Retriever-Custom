# Copyright (2026) Tsinghua University, Tencent Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adopted from https://github.com/bytedance/video-SALMONN-2. The original license is located at 'third-party-license/video-salmonn-2.txt'.
# Adopted from https://github.com/QwenLM/Qwen2.5-VL. The original license is located at 'third-party-license/qwenvl.txt'.

import os
import copy
import json
import random
import logging
import re
import time
import math
import itertools
import ast
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Tuple
from io import BytesIO
import base64
from collections.abc import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageOps
# from torchcodec.decoders import VideoDecoder, AudioDecoder
from decord import VideoReader, cpu
import decord
import soundfile as sf

# FFmpeg's h264 decoder logs "[h264 @ 0x...] mmco: unref short failure" at error
# level for many YouTube encodes (non-conforming reference-picture marking), most
# often when decord seeks into the middle of a video to reach an event window.
# The frames still decode, and a genuine decode failure raises in Python and is
# handled by __getitem__, so the log line is noise -- dozens per step in the
# training log. Set FFMPEG_VERBOSE=1 to see FFmpeg's messages again.
if os.environ.get("FFMPEG_VERBOSE", "0") != "1":
    try:
        decord.logging.set_level(decord.logging.QUIET)
    except AttributeError:  # decord build without the logging module
        pass
import librosa
try:
    import ffmpeg
except ImportError:
    ffmpeg = None
import transformers

import sys
if __name__ == "__main__":
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent
    sys.path.append(str(project_root))

from qwenvl.train.utils import IGNORE_INDEX, IMAGE_TOKEN_INDEX, VIDEO_TOKEN_INDEX, PAD_TOKEN_ID, DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN, DEFAULT_AUDIO_TOKEN

# The event audio cutter and the fixed-duration crop live in the inference
# package so training and inference cut an event's audio identically.
_SRC_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "src")
if os.path.isdir(_SRC_ROOT) and os.path.abspath(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, os.path.abspath(_SRC_ROOT))
from omniretriever.data.media import fit_waveform, load_audio_segment
from omniretriever.data.clip import load_clip

def is_rank_zero():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", None))
    if rank is not None:
        try:
            return int(rank) == 0
        except ValueError:
            pass
    return True

def rank0_print(*args, **kwargs):
    if is_rank_zero():
        kwargs.setdefault("flush", True)
        print(*args, **kwargs)

def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def resolve_media_path(filename: str, env_var: str) -> str:
    """Resolve a (possibly relative) media filename to an absolute path.

    If *filename* is already an absolute path it is returned unchanged.
    Otherwise the value of the environment variable *env_var* is used as a
    root directory and prepended to *filename*.

    Supported env vars:
      VIDEO_ROOT  – root directory that contains video files
      AUDIO_ROOT  – root directory that contains audio files
      IMAGE_ROOT  – root directory that contains image files

    Example::

        export VIDEO_ROOT=D:/KL/Data/YouCookII/YouCookII/videos
        export AUDIO_ROOT=D:/KL/Data/YouCookII/YouCookII/audio
    """
    if not filename:
        return filename
    if os.path.isabs(filename):
        return filename
    root = os.environ.get(env_var, "")
    if root:
        return os.path.join(root, filename)
    return filename

def split_into_groups(counts, groups):
    result = []
    for count, g in zip(counts, groups):
        base = count // g
        remainder = count % g
        group_list = [base + 1] * remainder + [base] * (g - remainder)
        result.append(group_list)
    return result


def augment_video_frames(video_np):
    """Apply random augmentations to video frames (numpy array, F x H x W x C, uint8)."""
    # Random horizontal flip
    if random.random() > 0.5:
        video_np = np.flip(video_np, axis=2).copy()
    # Random brightness jitter
    if random.random() > 0.5:
        factor = random.uniform(0.8, 1.2)
        video_np = np.clip(video_np.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    # Random contrast jitter
    if random.random() > 0.5:
        factor = random.uniform(0.8, 1.2)
        mean = video_np.mean(axis=(1, 2, 3), keepdims=True)
        video_np = np.clip((video_np.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)
    # Random saturation jitter (convert to float, adjust, clip)
    if random.random() > 0.5:
        factor = random.uniform(0.8, 1.2)
        gray = np.mean(video_np.astype(np.float32), axis=-1, keepdims=True)
        video_np = np.clip(gray + (video_np.astype(np.float32) - gray) * factor, 0, 255).astype(np.uint8)
    return video_np


def augment_audio_waveform(audio_wav):
    """Apply random augmentations to audio waveform (1D numpy array, float). Preserves length."""
    orig_len = len(audio_wav)
    orig_dtype = audio_wav.dtype
    audio_wav = audio_wav.astype(np.float64)
    # Volume perturbation
    if random.random() > 0.5:
        gain = random.uniform(0.8, 1.2)
        audio_wav = audio_wav * gain
    # Additive Gaussian noise
    if random.random() > 0.5:
        noise_level = random.uniform(0.001, 0.01) * np.abs(audio_wav).max()
        audio_wav = audio_wav + np.random.randn(orig_len) * noise_level
    # Speed perturbation (resample then truncate/pad to preserve length)
    if random.random() > 0.5:
        speed = random.uniform(0.9, 1.1)
        new_len = int(orig_len / speed)
        if new_len > 1:
            indices = np.linspace(0, orig_len - 1, new_len)
            audio_wav = np.interp(indices, np.arange(orig_len), audio_wav)
            if len(audio_wav) > orig_len:
                audio_wav = audio_wav[:orig_len]
            elif len(audio_wav) < orig_len:
                audio_wav = np.pad(audio_wav, (0, orig_len - len(audio_wav)))
    # Time shift (circular)
    if random.random() > 0.5:
        shift = int(random.uniform(-0.1, 0.1) * orig_len)
        audio_wav = np.roll(audio_wav, shift)
    return audio_wav.astype(orig_dtype)


def augment_text(text):
    """Apply random augmentations to text caption. Returns augmented text."""
    words = text.split()
    if len(words) <= 2:
        return text
    # Random word deletion (drop 10-20% of words)
    if random.random() > 0.5 and len(words) > 3:
        n_keep = max(2, int(len(words) * random.uniform(0.8, 0.9)))
        indices = sorted(random.sample(range(len(words)), n_keep))
        words = [words[i] for i in indices]
    # Random adjacent word swap (1-2 pairs)
    if random.random() > 0.5 and len(words) > 2:
        n_swaps = random.randint(1, min(2, len(words) // 2))
        for _ in range(n_swaps):
            i = random.randint(0, len(words) - 2)
            words[i], words[i + 1] = words[i + 1], words[i]
    return ' '.join(words)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_args):
        super(LazySupervisedDataset, self).__init__()

        dataset = data_args.dataset_use.split(",")
        dataset_list = dataset
        rank0_print(f"Loading datasets: {dataset_list}")
        self.video_max_total_pixels = getattr(
            data_args, "video_max_total_pixels", 1664 * 28 * 28
        )
        self.video_min_total_pixels = getattr(
            data_args, "video_min_total_pixels", 256 * 28 * 28
        )
        self.data_args = data_args
        
        self.omni_processor = data_args.omni_processor
        self.data_args.image_processor = self.omni_processor.image_processor
        self.data_args.audio_processor = self.omni_processor.feature_extractor

        list_data_dict = []

        for data in dataset_list:
            file_format = data.split(".")[-1]
            if file_format == "jsonl":
                annotations = read_jsonl(data)
            else:
                annotations = json.load(open(data, "r"))
            list_data_dict += annotations

        for d in list_data_dict:
            if d["conversations"][0]["from"] == "system":
                idx = 1
            else:
                idx = 0
            if "<image>" in d["conversations"][idx]["value"] and not "image" in d and ("video" in d or "frame_dir" in d):
                d["conversations"][idx]["value"] = d["conversations"][idx]["value"].replace(
                    "<image>", "<video>"
                )
            if "<image>" in d["conversations"][idx]["value"] and not "image" in d and not ("video" in d or "frame_dir" in d) and ("audio" in d):
                d["conversations"][idx]["value"] = d["conversations"][idx]["value"].replace(
                    "<image>", "<audio>"
                )

        # Optional video blacklist (set VIDEO_BLACKLIST=<path> to skip ids in that file).
        blacklist_path = os.environ.get("VIDEO_BLACKLIST", "")
        if blacklist_path:
            rank0_print(f"Using blacklist: {blacklist_path}")
        if blacklist_path and os.path.isfile(blacklist_path):
            with open(blacklist_path) as f:
                blacklist = set(line.strip() for line in f if line.strip())
            before = len(list_data_dict)
            list_data_dict = [d for d in list_data_dict if d.get("video", "") not in blacklist]
            rank0_print(f"Blacklist filtered: {before} → {len(list_data_dict)} (removed {before - len(list_data_dict)} samples)")

        rank0_print(f"Total training samples: {len(list_data_dict)}")

        # random.shuffle(list_data_dict, seed=2025)  # Randomly shuffle the data for training

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        # self.data_args = data_args
        self.data_args.image_processor.max_pixels = data_args.max_pixels
        self.data_args.image_processor.min_pixels = data_args.min_pixels
        self.data_args.image_processor.size["longest_edge"] = data_args.max_pixels
        self.data_args.image_processor.size["shortest_edge"] = data_args.min_pixels

        self.type_list = [it.get("type", "retrieval") for it in list_data_dict]
        self.type_dict = {}
        for k, it in enumerate(list_data_dict):
            tp = it.get("type", "retrieval")
            if tp not in self.type_dict:
                self.type_dict[tp] = [k]
            else:
                self.type_dict[tp].append(k)

    def __len__(self):
        return len(self.list_data_dict)

    def process_audio(self, audio_file=None, audio_wav=None, timestamps=None):
        try:
            audio_kwargs = {'sampling_rate': 16000, 'padding': 'max_length', 'return_attention_mask': True, 'return_tensors': 'pt'}

            processor = self.data_args.audio_processor

            mel = []
            if audio_wav is None:
                if isinstance(audio_file, list):
                    audio_data = []
                    for file in audio_file:
                        audio, sr = sf.read(file)
                        if len(audio.shape) == 2:
                            audio = audio[:, 0]
                        if sr != 16000:
                            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                        audio_data.append(audio)
                else:
                    audio, sr = sf.read(audio_file)
                    if len(audio.shape) == 2:
                        audio = audio[:, 0]
                    if sr != 16000:
                        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
                    if timestamps is not None:
                        start = int(timestamps[0] * 16000)
                        # Only clip when the window actually lies inside the file.
                        # YouCookII ships one .wav per event, so the file is already
                        # the segment while its timestamps address the full video and
                        # point past the end -- clipping by them empties the waveform,
                        # and an empty waveform is dropped silently further down.
                        # A manifest whose audio is the full track still gets clipped.
                        if start < len(audio):
                            audio = audio[start: int(timestamps[1] * 16000)]
                    audio_data = [audio]
            else:
                sr = 16000
                audio_data = [audio_wav]

            audio_inputs = []
            audio_lengths = []
            for idx in range(len(audio_data)):
                feature_attention_mask_idx = []
                input_features_idx = []
                audio_lst = [audio_data[idx][k: k + 300 * audio_kwargs["sampling_rate"]] for k in range(0, len(audio_data[idx]), 300 * audio_kwargs["sampling_rate"])]
                audio_lengths_seg = 0
                for audio_seg in audio_lst:
                    if audio_seg.shape[0] < audio_kwargs["sampling_rate"]:
                        padding = audio_kwargs["sampling_rate"] - audio_seg.shape[0]
                        audio_seg = np.pad(audio_seg, (0, padding), mode="constant", constant_values=0)
                    audio_inputs_seg = self.data_args.audio_processor(audio_seg, **audio_kwargs)
                    attn_seg = audio_inputs_seg.pop("attention_mask")
                    feature_attention_mask_idx.append(attn_seg)
                    input_features_idx.append(audio_inputs_seg.pop("input_features"))
                    input_lengths_seg = (attn_seg.sum(-1) - 1) // 2 + 1
                    audio_lengths_seg += (input_lengths_seg - 2) // 2 + 1
                
                if audio_lengths_seg <= 0:
                    # Returning None here drops the audio for this record without
                    # raising, which also strips it of its TupleInfoNCE data and
                    # zeroes obj1 for every batch it lands in. Say so.
                    rank0_print(
                        f"[audio] empty waveform: file={audio_file} timestamps={timestamps}"
                        " -- record trains without audio."
                    )
                    return None, None, None

                feature_attention_mask_idx = torch.cat(feature_attention_mask_idx, dim=0)
                input_features_idx = torch.cat(input_features_idx, dim=0)

                audio_inputs.append({
                    "feature_attention_mask": feature_attention_mask_idx,
                    "input_features": input_features_idx
                })
                audio_lengths.append(audio_lengths_seg)

            return audio_inputs, audio_lengths, audio_data
        
        except Exception as e:
            print(f"Process Audio Error: {e},  file: {audio_file}, line: {e.__traceback__.tb_lineno}")
            raise e
            

    def process_image_unified(self, image_file):
        processor = copy.deepcopy(self.data_args.image_processor)
        processor.max_pixels = self.data_args.image_max_frame_pixels
        processor.min_pixels = self.data_args.image_min_frame_pixels
        processor.size["longest_edge"] = processor.max_pixels
        processor.size["shortest_edge"] = processor.min_pixels

        image = Image.open(image_file).convert("RGB")
        width, height = image.size

        if width >= 28 and height >= 28:
            visual_processed = processor.preprocess(image, return_tensors="pt")
        else:
            pad_width = max(0, 28 - width)
            pad_height = max(0, 28 - height)
            
            if pad_width == 0 and pad_height == 0:
                pass
            else:
                left = pad_width // 2
                right = pad_width - left
                top = pad_height // 2
                bottom = pad_height - top
                fill_color=(0, 0, 0)
                padded_image = ImageOps.expand(image, border=(left, top, right, bottom), fill=fill_color)

            visual_processed = processor.preprocess(padded_image, return_tensors="pt")

        image_tensor = visual_processed["pixel_values"]
        if isinstance(image_tensor, List):
            image_tensor = image_tensor[0]
        grid_thw = visual_processed["image_grid_thw"][0]
        return image_tensor, grid_thw

    def process_video(self, video_file, timestamps=None, max_frame_num=-1):
        t0 = time.time()
        video = self.video_decord(video_file, timestamps=timestamps, max_frame_num=max_frame_num)
        elapsed = time.time() - t0
        if elapsed > 30:
            print(f"[WARN] Slow video decode: {video_file} took {elapsed:.1f}s")
        return video

    def video_decord_augmented(self, video_file, timestamps=None, max_frame_num=-1):
        """Decode video with data augmentation (same frame sampling, augmented pixels)."""
        self._check_video_file(video_file)
        vr = VideoReader(video_file, num_threads=1)
        total_frame_num = len(vr)
        video_length = total_frame_num / vr.get_avg_fps()
        video_min_frames = getattr(self.data_args, "video_min_frames", 1)
        video_max_frames = getattr(self.data_args, "video_max_frames", 600) if max_frame_num <= 0 else max_frame_num
        interval = getattr(self.data_args, "base_interval", 0.5)

        if timestamps is None:
            num_frames_to_sample = round(video_length / interval)
            target_frames = min(max(num_frames_to_sample, video_min_frames), video_max_frames)
            frame_idx = np.linspace(0, total_frame_num - 1, target_frames, dtype=int)
        else:
            video_length = min(timestamps[1], video_length) - timestamps[0]
            num_frames_to_sample = round(video_length / interval)
            target_frames = min(max(num_frames_to_sample, video_min_frames), video_max_frames)
            start_idx = round(timestamps[0] * vr.get_avg_fps())
            end_idx = min(round(timestamps[1] * vr.get_avg_fps()), total_frame_num - 1)
            if video_length <= 0 or start_idx > end_idx:
                # The event starts after the file ends (a truncated download);
                # linspace would run backwards into out-of-bound indices.
                raise ValueError(f"Event {timestamps} lies past the end of {video_file} "
                                 f"({total_frame_num / vr.get_avg_fps():.1f}s)")
            frame_idx = np.linspace(start_idx, end_idx, target_frames, dtype=int)

        video = vr.get_batch(frame_idx).asnumpy()
        video = augment_video_frames(np.array(video))
        video = torch.from_numpy(video)
        video_proc = self.data_args.image_processor(images=None, videos=video, return_tensors="pt")
        fps = len(frame_idx) / video_length
        fps = [fps] * 1
        video_proc["video_second_per_grid"] = [self.data_args.image_processor.temporal_patch_size / fps[i] for i in range(len(fps))]
        return video_proc["pixel_values_videos"], video_proc['video_grid_thw'], video_proc["video_second_per_grid"]

    def _check_video_file(self, video_file):
        """Quick sanity check before decord — raise on bad files."""
        if not os.path.isfile(video_file):
            raise FileNotFoundError(f"Video not found: {video_file}")
        fsize = os.path.getsize(video_file)
        if fsize < 1024:
            raise ValueError(f"Video too small ({fsize}B), likely corrupted: {video_file}")
        # MAX_VIDEO_MB: HT-Step AV1 videos converted to lossless H.264 (scripts/download_htstep.py) pass 500 MB
        if fsize > int(os.environ.get("MAX_VIDEO_MB", 500)) * 1024 * 1024:
            raise ValueError(f"Video too large ({fsize // 1024 // 1024}MB), skipping: {video_file}")

    @staticmethod
    def _video_decode_timeout_handler(signum, frame):
        raise TimeoutError("Video decode timed out (>30s)")

    def video_decord(self, video_file, timestamps=None, max_frame_num=-1):
        self._check_video_file(video_file)
        vr = VideoReader(video_file, num_threads=1) # , ctx=cpu(0)
        total_frame_num = len(vr)

        video_length = total_frame_num / vr.get_avg_fps()
        video_min_frames = getattr(self.data_args, "video_min_frames", 1)
        video_max_frames = getattr(self.data_args, "video_max_frames", 600) if max_frame_num <= 0 else max_frame_num

        interval = getattr(self.data_args, "base_interval", 0.5)

        if timestamps is None:
            num_frames_to_sample = round(video_length / interval)
            target_frames = min(max(num_frames_to_sample, video_min_frames), video_max_frames)
            frame_idx = np.linspace(0, total_frame_num - 1, target_frames, dtype=int)
        else:
            video_length = min(timestamps[1], video_length) - timestamps[0]
            num_frames_to_sample = round(video_length / interval)
            target_frames = min(max(num_frames_to_sample, video_min_frames), video_max_frames)

            start_idx = round(timestamps[0] * vr.get_avg_fps())
            end_idx = min(round(timestamps[1] * vr.get_avg_fps()), total_frame_num - 1)
            if video_length <= 0 or start_idx > end_idx:
                # The event starts after the file ends (a truncated download);
                # linspace would run backwards into out-of-bound indices.
                raise ValueError(f"Event {timestamps} lies past the end of {video_file} "
                                 f"({total_frame_num / vr.get_avg_fps():.1f}s)")
            frame_idx = np.linspace(start_idx, end_idx, target_frames, dtype=int)

        video = vr.get_batch(frame_idx).asnumpy() # video: (F, H, W, C)

        video = np.array(video)

        video = torch.from_numpy(video)
        video_proc = self.data_args.image_processor(images=None, videos=video, return_tensors="pt")
        fps = len(frame_idx) / video_length     # 1 / interval
        fps = [fps] * 1
        video_proc["video_second_per_grid"] = [self.data_args.image_processor.temporal_patch_size / fps[i] for i in range(len(fps))]
        return video_proc["pixel_values_videos"], video_proc['video_grid_thw'], video_proc["video_second_per_grid"]

    def process_clip(self, clip_file):
        """A pre-cut clip (scripts/precut_htstep.py): the frames video_decord would pick, already resized by the
        processor's own resize, so the processor runs with do_resize=False. Returns video_decord's outputs and the
        event audio (None = the video has no audio)."""
        clip = load_clip(clip_file)
        video_proc = self.data_args.image_processor(
            images=None, videos=torch.from_numpy(clip["frames"]), do_resize=False, return_tensors="pt")
        fps = clip["n_frames"] / clip["video_length"]
        second_per_grid = [self.data_args.image_processor.temporal_patch_size / fps]
        return video_proc["pixel_values_videos"], video_proc['video_grid_thw'], second_per_grid, clip["wav"]

    def process_omni_conversations(self, conversations, type, gpt="gpt"):
        label = None
        omni_conversations = []
        for conv in conversations:
            if conv["from"] == "human":
                if "<video>\n" in conv["value"]:
                    omni_conversations.append({
                        "role": "user",
                        "content": [{"type": "video"}, {"type": "text", "text": conv["value"].replace("<video>\n", "")}]
                    })
                elif "<audio>\n" in conv["value"]:
                    omni_conversations.append({
                        "role": "user",
                        "content": [{"type": "audio"}, {"type": "text", "text": conv["value"].replace("<audio>\n", "")}]
                    })
                elif "<image>\n" in conv["value"]:
                    omni_conversations.append({
                        "role": "user",
                        "content": [{"type": "image"}, {"type": "text", "text": conv["value"].replace("<image>\n", "")}]
                    })
                else:
                    omni_conversations.append({
                        "role": "user",
                        "content": [{"type": "text", "text": conv["value"]}]
                    })
            elif conv["from"] == gpt:
                if "sft" not in type and self.data_args.train_classify:
                    label = conv["value"]
                else:
                    omni_conversations.append({
                        "role": "assistant",
                        "content": [{"type": "text", "text": conv["value"]}]
                    })
        
        text = self.omni_processor.apply_chat_template(omni_conversations, add_generation_prompt=False, tokenize=False)

        if "sft" not in type and self.data_args.train_classify:
            text = [text[0].split("<|im_start|>user\n")[-1].strip()]
        else:
            text = [text[0].strip()]

        return text, label

    def gen_omni_labels(self, text):
        all_convs = text.split("<|im_start|>")
        im_start = self.tokenizer("<|im_start|>")["input_ids"]
        labels = []
        for conv in all_convs:
            if conv:
                if conv.startswith("system\n") or conv.startswith("user\n"):
                    labels += [IGNORE_INDEX] * (self.tokenizer(conv, padding=True, padding_side="left", return_tensors="pt")["input_ids"].size(1) + 1)
                elif conv.startswith("assistant\n"):
                    labels += [IGNORE_INDEX] * 3    # <|im_start|>assistant\n
                    labels += self.tokenizer(conv[len("assistant\n"):], padding=True, padding_side="left", return_tensors="pt")["input_ids"].tolist()[0]
                else:
                    raise NotImplementedError
        
        labels = torch.tensor(labels, dtype=torch.long).unsqueeze(0)
        return labels

    def _prepare_submodal_input(self, modalities, label_text,
                               audio_lengths_0, video_grid_thw_0, second_per_grid_ts):
        """
        Prepare tokenized input_ids for a sub-modal combination.
        Matches original data format: "<image>\\nPlease describe the video."
          - When video present: audio injected via use_audio_in_video (NOT separate <audio> tag)
          - When only audio (no video): uses <audio> tag
          - 'text' in modalities → caption replaces "Please describe the video."

        Args:
            modalities: set of strings, e.g. {'audio'}, {'video', 'text'}, etc.
            label_text: the text caption (used when 'text' is in modalities)
            audio_lengths_0: audio_lengths[0] tensor (for replace_multimodal_special_tokens)
            video_grid_thw_0: video_grid_thw[0] tensor
            second_per_grid_ts: list of second_per_grid values
        Returns:
            input_ids tensor [1, seq_len]
        """
        has_audio = 'audio' in modalities
        has_video = 'video' in modalities
        use_audio_in_video = has_audio and has_video

        # Build content items matching original format:
        # - video+audio: only {"type":"video"}, audio injected via use_audio_in_video
        # - audio only:  {"type":"audio"} as separate tag
        # - video only:  {"type":"video"}
        content = []
        if has_video:
            content.append({"type": "video"})
            # audio NOT added as content item; injected via use_audio_in_video
        elif has_audio:
            content.append({"type": "audio"})

        # Text part: caption when 'text' in modalities, otherwise original prompt
        if 'text' in modalities:
            content.append({"type": "text", "text": label_text})
        else:
            content.append({"type": "text", "text": "Please describe the video."})

        omni_conv = [{"role": "user", "content": content}]
        text = self.omni_processor.apply_chat_template(
            omni_conv, add_generation_prompt=False, tokenize=False
        )
        text = [text[0].split("<|im_start|>user\n")[-1].strip()]

        text = self.omni_processor.replace_multimodal_special_tokens(
            text,
            iter(audio_lengths_0) if has_audio else iter([]),
            iter([]),  # no image
            iter(video_grid_thw_0) if has_video else iter([]),
            video_second_per_grid=iter(second_per_grid_ts) if has_video else iter([]),
            use_audio_in_video=use_audio_in_video,
            position_id_per_seconds=25,
            seconds_per_chunk=2.0 * second_per_grid_ts[0] if has_video and second_per_grid_ts else None,
        )

        if has_audio and self.data_args.use_beats and not self.data_args.beats_only:
            text[0] = text[0].replace("<|AUDIO|>", "<|AUDIO|><|AUDIO|>")

        token_res = self.tokenizer(text, padding=True, padding_side="left", return_tensors="pt")
        return token_res["input_ids"]

    @staticmethod
    def _getitem_timeout_handler(signum, frame):
        raise TimeoutError(f"Sample loading timed out (>30s) in worker {os.getpid()}")

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        import signal
        _HAS_SIGALRM = hasattr(signal, "SIGALRM")  # False on Windows
        if _HAS_SIGALRM:
            old_handler = signal.signal(signal.SIGALRM, self._getitem_timeout_handler)
            signal.alarm(30)
        try:
            sample = self._get_item(i)
            if _HAS_SIGALRM:
                signal.alarm(0)
        except Exception as e:
            if _HAS_SIGALRM:
                signal.alarm(0)
            print(f"[{'TIMEOUT' if isinstance(e, TimeoutError) else 'ERROR'}] Sample {i} failed: {e}, replacing", flush=True)
            # Try a random replacement sample
            for _retry in range(3):
                try:
                    randidx = random.choice(self.type_dict.get(
                        self.list_data_dict[i].get("type", "retrieval"), list(range(len(self.list_data_dict)))))
                    if _HAS_SIGALRM:
                        signal.alarm(30)
                    sample = self._get_item(randidx)
                    if _HAS_SIGALRM:
                        signal.alarm(0)
                    break
                except Exception:
                    if _HAS_SIGALRM:
                        signal.alarm(0)
                    continue
            else:
                # All retries failed — return first sample as last resort
                if _HAS_SIGALRM:
                    signal.alarm(0)
                sample = self._get_item(0)
        finally:
            if _HAS_SIGALRM:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
        return sample

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        try:
            sources = self.list_data_dict[i]
            if isinstance(i, int):
                sources = [sources]
            assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

            # Debug: log which file is being loaded to a file (helps identify hang sources)
            _vid = sources[0].get("video", "")
            if _vid:
                _log_path = os.environ.get("DATA_LOADING_LOG", "./tmp/data_loading.log")
                try:
                    os.makedirs(os.path.dirname(os.path.abspath(_log_path)), exist_ok=True)
                    with open(_log_path, "a") as _logf:
                        _logf.write(f"{os.getpid()} {i} {_vid}\n")
                except Exception:
                    pass  # Non-critical logging, ignore errors

            # define some variables
            image = None
            grid_thw_merged = None
            video_grid_thw_merged = None
            grid_thw = None
            video = None
            video_grid_thw = None
            second_per_grid_ts = None
            audio = None
            audio_lengths = None
            raw_wav = None
            video_file = None

            if "image" in sources[0]:
                image_file = resolve_media_path(
                    self.list_data_dict[i]["image"], "IMAGE_ROOT"
                )
                image, grid_thw = self.process_image_unified(image_file)
                grid_thw = grid_thw.unsqueeze(0)
                image = [image]
                grid_thw_merged = copy.deepcopy(grid_thw)
                if not isinstance(grid_thw, Sequence):
                    grid_thw_merged = [grid_thw_merged]
                    grid_thw = [grid_thw]
                grid_thw_merged = [
                    merged_thw.prod() // self.data_args.image_processor.merge_size**2
                    for merged_thw in grid_thw_merged
                ]

            if "frame_dir" in sources[0]:
                video_max_frames = getattr(self.data_args, "video_max_frames", 600)
                video_file = resolve_media_path(
                    sources[0]["frame_dir"], "VIDEO_ROOT"
                )
                frame_files = os.listdir(video_file)
                frame_files = sorted(frame_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
                frame_files = [os.path.join(video_file, fr) for fr in frame_files]
                image_list = []
                for f_file in frame_files:
                    image_list.append(np.array(Image.open(f_file)))
                
                if len(image_list) > video_max_frames:
                    i_indices = np.linspace(0, len(image_list) - 1, video_max_frames, dtype=int).tolist()
                    image_list = [image_list[k] for k in i_indices]
                    
                video = np.stack(image_list)

                video = torch.from_numpy(video)
                video_proc = self.data_args.image_processor(images=None, videos=video, return_tensors="pt")
                fps = [len(image_list) / sources[0]["duration"] if "duration" in sources[0] and sources[0]["duration"] > 1 else 1] * 1
                video_proc["video_second_per_grid"] = [self.data_args.image_processor.temporal_patch_size / fps[i] for i in range(len(fps))]
                video, video_grid_thw, second_per_grid_ts = video_proc["pixel_values_videos"], video_proc['video_grid_thw'], video_proc["video_second_per_grid"]

                video = [video]

                video_grid_thw_merged = copy.deepcopy(video_grid_thw)
                if not isinstance(video_grid_thw, Sequence):
                    video_grid_thw_merged = [video_grid_thw_merged]
                    video_grid_thw = [video_grid_thw]
                else:
                    raise NotImplementedError

            elif "clip" in sources[0]:
                clip_file = resolve_media_path(sources[0]["clip"], "CLIP_ROOT")
                video, video_grid_thw, second_per_grid_ts, clip_wav = self.process_clip(clip_file)
                video = [video]
                video_grid_thw_merged = [copy.deepcopy(video_grid_thw)]
                video_grid_thw = [video_grid_thw]

            elif "video" in sources[0]:
                _raw_video = sources[0]["video"]
                timestamps = sources[0].get("timestamps", None)
                if isinstance(_raw_video, List):
                    video_file = [
                        resolve_media_path(f, "VIDEO_ROOT") for f in _raw_video
                    ]
                    if len(video_file) > 1:
                        results = [self.process_video(file) for file in video_file]
                        video, video_grid_thw, second_per_grid_ts = zip(*results)
                    else:
                        video_file = video_file[0]
                        video, video_grid_thw, second_per_grid_ts = self.process_video(video_file)
                        video = [video]
                else:
                    video_file = resolve_media_path(_raw_video, "VIDEO_ROOT")
                    video, video_grid_thw, second_per_grid_ts = self.process_video(video_file, timestamps=timestamps)
                    video = [video]

                video_grid_thw_merged = copy.deepcopy(video_grid_thw)
                if not isinstance(video_grid_thw, Sequence):
                    video_grid_thw_merged = [video_grid_thw_merged]
                    video_grid_thw = [video_grid_thw]

            # Fixed audio duration (in seconds) for consistent batching
            fixed_audio_duration = getattr(self.data_args, 'fixed_audio_duration', 0)

            if "audio" in sources[0]:
                _raw_audio = sources[0]["audio"]
                if isinstance(_raw_audio, list):
                    audio_file = [
                        resolve_media_path(f, "AUDIO_ROOT") for f in _raw_audio
                    ]
                else:
                    audio_file = resolve_media_path(_raw_audio, "AUDIO_ROOT")
                timestamps = sources[0].get("timestamps", None)
                try:
                    audio, audio_lengths, raw_wav = self.process_audio(audio_file, timestamps=timestamps)
                except Exception:
                    # Audio file unreadable, fall back to silence
                    target_len = int(max(fixed_audio_duration, 1) * 16000)
                    silence = np.zeros(target_len, dtype=np.float32)
                    audio, audio_lengths, raw_wav = self.process_audio(audio_wav=silence)
            elif "clip" in sources[0]:
                # Same as the video-only branch below: the clip holds the event audio already cut, None when the
                # cut failed (no audio stream), which that branch also turns into silence.
                if clip_wav is not None and len(clip_wav):
                    audio, audio_lengths, raw_wav = self.process_audio(audio_wav=clip_wav)
                else:
                    target_len = int(max(fixed_audio_duration, 1) * 16000)
                    audio, audio_lengths, raw_wav = self.process_audio(audio_wav=np.zeros(target_len, dtype=np.float32))
            elif isinstance(video_file, str) and "video" in sources[0]:
                # Video-only record: the event's audio is cut out of the same
                # video over the same [start, end] window as its frames.
                timestamps = sources[0].get("timestamps", None)
                try:
                    event_wav = load_audio_segment(video_file, *(timestamps or (None, None)))
                    audio, audio_lengths, raw_wav = self.process_audio(audio_wav=event_wav)
                except Exception as e:
                    rank0_print(f"[audio] cannot cut audio from {video_file} {timestamps}: {e} -- using silence.")
                    target_len = int(max(fixed_audio_duration, 1) * 16000)
                    silence = np.zeros(target_len, dtype=np.float32)
                    audio, audio_lengths, raw_wav = self.process_audio(audio_wav=silence)
            elif "video" in sources[0] and getattr(self.data_args, 'use_beats', False):
                # For samples without audio but with video, generate silence
                target_len = int(max(fixed_audio_duration, 1) * 16000)
                silence = np.zeros(target_len, dtype=np.float32)
                audio, audio_lengths, raw_wav = self.process_audio(audio_wav=silence)

            # Crop/pad to a fixed length for consistent batching. Centre crop,
            # the same fit_waveform the inference loader applies.
            if fixed_audio_duration > 0 and raw_wav is not None:
                target_len = int(fixed_audio_duration * 16000)
                if len(raw_wav) != 1 or len(raw_wav[0]) != target_len:
                    # Reprocess audio with fixed length for consistent audio_lengths
                    audio, audio_lengths, raw_wav = self.process_audio(
                        audio_wav=fit_waveform(raw_wav[0], target_len))

            if raw_wav is not None and len(raw_wav[0]) < 16000: # pad audio to at least 1s
                sil = np.zeros(16000 - len(raw_wav[0]), dtype=float)
                raw_wav[0] = np.concatenate((raw_wav[0], sil), axis=0)
            
            chat_sources = copy.deepcopy([e["conversations"] for e in sources])

            text, label = self.process_omni_conversations(sources[0]["conversations"], sources[0].get("type", "retrieval"), gpt="gpt")
            text = self.omni_processor.replace_multimodal_special_tokens(
                text,
                iter(audio_lengths[0]) if audio is not None else iter([]),
                iter(grid_thw[0]) if image is not None else iter([]),
                iter(video_grid_thw[0]) if video_grid_thw is not None else iter([]),
                video_second_per_grid=iter(second_per_grid_ts) if video_grid_thw is not None else iter([]),
                use_audio_in_video=audio is not None,
                position_id_per_seconds=25,
                seconds_per_chunk=2.0 * second_per_grid_ts[0] if second_per_grid_ts is not None else None,
            )
            assert len(text) == 1
            if (self.data_args.use_beats and not self.data_args.beats_only):
                text[0] = text[0].replace("<|AUDIO|>", "<|AUDIO|><|AUDIO|>")
            
            labels = None
            if "sft" in sources[0].get("type", "retrieval") or not self.data_args.train_classify:
                labels = self.gen_omni_labels(text[0])

            token_res = self.tokenizer(text, padding=True, padding_side="left", return_tensors="pt")
            input_ids = token_res["input_ids"]
            attention_mask = token_res["attention_mask"]

            label_ids = None
            if sources[0].get("type", "retrieval") != "sft":
                if label:
                    if not label.endswith("<|im_end|>"):
                        label += "<|im_end|>"
                    label_ids = self.tokenizer(label, padding=True, padding_side="left", return_tensors="pt")["input_ids"]

            
            if "pos_image" in sources[0]:
                if "image" in sources[0]["pos_image"]:
                    pos_image_file = resolve_media_path(sources[0]["pos_image"]["image"], "IMAGE_ROOT")
                    pos_image, pos_grid_thw = self.process_image_unified(pos_image_file)
                    pos_image = [pos_image]
                    pos_grid_thw = [pos_grid_thw.unsqueeze(0)]
                else:
                    pos_image, pos_grid_thw = None, None

                if "video" in sources[0]["pos_image"]:
                    pos_video_file = resolve_media_path(sources[0]["pos_image"]["video"], "VIDEO_ROOT")
                    pos_video, pos_video_grid_thw, pos_second_per_grid_ts = self.process_video(pos_video_file)
                    pos_video = [pos_video]
                    pos_video_grid_thw = [pos_video_grid_thw]
                else:
                    pos_video, pos_video_grid_thw, pos_second_per_grid_ts = None, None, None

                pos_text, _ = self.process_omni_conversations(sources[0]["pos_image"]["conversations"], sources[0].get("type", "retrieval"), gpt="gpt")
                pos_text = self.omni_processor.replace_multimodal_special_tokens(
                    pos_text,
                    iter([]),
                    iter(pos_grid_thw[0]) if pos_grid_thw is not None else iter([]),
                    iter(pos_video_grid_thw[0]) if pos_video_grid_thw is not None else iter([]),
                    video_second_per_grid=iter(pos_second_per_grid_ts) if pos_second_per_grid_ts is not None else iter([]),
                    use_audio_in_video=False,
                    position_id_per_seconds=25,
                    seconds_per_chunk=None,
                )
                pos_token_res = self.tokenizer(pos_text, padding=True, padding_side="left", return_tensors="pt")
                pos_input_ids = pos_token_res["input_ids"]
                pos_attention_mask = pos_token_res["attention_mask"]

            else:
                pos_video, pos_video_grid_thw, pos_second_per_grid_ts = None, None, None
                pos_image, pos_grid_thw, pos_input_ids, pos_attention_mask = None, None, None, None


            data_dict = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": image[0] if image is not None else None,
                "image_grid_thw": grid_thw[0] if grid_thw is not None else None,
                "pixel_values_videos": video[0] if video is not None else None,
                "video_grid_thw": video_grid_thw[0] if video_grid_thw is not None else None,
                "video_second_per_grid": second_per_grid_ts[0] if second_per_grid_ts is not None else None,
                "input_features": audio[0]["input_features"] if audio is not None else None,
                "feature_attention_mask": audio[0]["feature_attention_mask"] if audio is not None else None,
                "label_ids": label_ids,
                "type": sources[0].get("type", "retrieval"),
                "labels": labels,
                "pos_pixel_values": pos_image[0] if pos_image is not None else None,
                "pos_image_grid_thw": pos_grid_thw[0] if pos_grid_thw is not None else None,
                "pos_input_ids": pos_input_ids,
                "pos_attention_mask": pos_attention_mask,
                "pos_pixel_values_videos": pos_video[0] if pos_video is not None else None,
                "pos_video_grid_thw": pos_video_grid_thw[0] if pos_video_grid_thw is not None else None,
                "pos_video_second_per_grid": pos_second_per_grid_ts,
                "input_raw_wav": torch.from_numpy(raw_wav[0]) if raw_wav is not None else None,
                # carried to the loss for the same-text negative mask (trainer --mask_same_text)
                "text": sources[0].get("text"),
                "id": sources[0].get("id"),
            }

            # TupleInfoNCE: augmented video/audio/text + sub-modal inputs
            _tuple_ok = False
            if getattr(self.data_args, 'use_tuple_infonce', False) and video is not None and audio is not None and label:
              try:
                has_video = video_grid_thw is not None
                has_audio = audio is not None

                # --- Augmented video (positive pair) ---
                video_file_for_aug = sources[0].get("video", None)
                if video_file_for_aug is not None:
                    timestamps_aug = sources[0].get("timestamps", None)
                    if isinstance(video_file_for_aug, list):
                        video_file_for_aug = video_file_for_aug[0] if len(video_file_for_aug) == 1 else video_file_for_aug
                    if isinstance(video_file_for_aug, str):
                        video_file_for_aug = resolve_media_path(video_file_for_aug, "VIDEO_ROOT")
                        aug_vid, aug_vgt, aug_spg = self.video_decord_augmented(video_file_for_aug, timestamps=timestamps_aug)
                        data_dict["aug_pixel_values_videos"] = aug_vid
                        data_dict["aug_video_grid_thw"] = aug_vgt
                        data_dict["aug_video_second_per_grid"] = aug_spg[0]
                    else:
                        data_dict["aug_pixel_values_videos"] = None
                        data_dict["aug_video_grid_thw"] = None
                        data_dict["aug_video_second_per_grid"] = None
                else:
                    data_dict["aug_pixel_values_videos"] = None
                    data_dict["aug_video_grid_thw"] = None
                    data_dict["aug_video_second_per_grid"] = None

                # --- Augmented audio (positive pair) ---
                if raw_wav is not None:
                    aug_wav = augment_audio_waveform(raw_wav[0].copy())
                    aug_audio, aug_audio_lengths, aug_raw_wav = self.process_audio(audio_wav=aug_wav)
                    data_dict["aug_input_features"] = aug_audio[0]["input_features"]
                    data_dict["aug_feature_attention_mask"] = aug_audio[0]["feature_attention_mask"]
                    data_dict["aug_input_raw_wav"] = torch.from_numpy(aug_raw_wav[0])
                else:
                    data_dict["aug_input_features"] = None
                    data_dict["aug_feature_attention_mask"] = None
                    data_dict["aug_input_raw_wav"] = None

                # --- Augmented text (only for Objective 1 positive pair) ---
                aug_label = augment_text(label)

                # Sub-modal input_ids
                al0 = audio_lengths[0] if audio_lengths is not None else None
                vgt0 = video_grid_thw[0] if video_grid_thw is not None else None
                spg = second_per_grid_ts if second_per_grid_ts is not None else None

                # Anchor input_ids: A+V+T where T = original caption
                anchor_ids = self._prepare_submodal_input(
                    {'audio', 'video', 'text'}, label, al0, vgt0, spg)
                data_dict["input_ids"] = anchor_ids
                data_dict["attention_mask"] = anchor_ids.ne(PAD_TOKEN_ID).to(torch.int64)

                # Positive pair input_ids: A'+V'+T' (all augmented, Objective 1 only)
                data_dict["tuple_pos_input_ids"] = self._prepare_submodal_input(
                    {'audio', 'video', 'text'}, aug_label, al0, vgt0, spg)

                # Hard negative: shuffle text (current A_i,V_i + random other T_j)
                neg_idx = random.choice(self.type_dict.get(sources[0].get("type", "retrieval"), list(range(len(self.list_data_dict)))))
                while neg_idx == i:
                    neg_idx = random.choice(self.type_dict.get(sources[0].get("type", "retrieval"), list(range(len(self.list_data_dict)))))
                neg_caption = self.list_data_dict[neg_idx]["conversations"][-1]["value"]
                data_dict["neg_text_input_ids"] = self._prepare_submodal_input(
                    {'audio', 'video', 'text'}, neg_caption, al0, vgt0, spg)

                # --- Objective 2&3: all use ORIGINAL caption/audio/video ---
                # Audio-only (original A + generic prompt)
                data_dict["audio_only_input_ids"] = self._prepare_submodal_input(
                    {'audio'}, label, al0, vgt0, spg)
                # Video-only (original V + generic prompt)
                data_dict["video_only_input_ids"] = self._prepare_submodal_input(
                    {'video'}, label, al0, vgt0, spg)
                # Text + Audio (original caption + original A)
                data_dict["ta_input_ids"] = self._prepare_submodal_input(
                    {'audio', 'text'}, label, al0, vgt0, spg)
                # Text + Video (original caption + original V)
                data_dict["tv_input_ids"] = self._prepare_submodal_input(
                    {'video', 'text'}, label, al0, vgt0, spg)
                # Audio + Video (original A+V + generic prompt)
                data_dict["av_input_ids"] = self._prepare_submodal_input(
                    {'audio', 'video'}, label, al0, vgt0, spg)
                _tuple_ok = True
              except Exception as e:
                print(f"TupleInfoNCE data prep failed (sample {i}): {e}, line: {e.__traceback__.tb_lineno}, falling back")
                _tuple_ok = False

            if not _tuple_ok:
                # The collator drops a field for the whole batch when any one
                # sample is missing it, so a rare gap here zeroes obj1 for a
                # whole optimiser step. Name the first one rather than let the
                # objective quietly read zero for a full run.
                if (getattr(self.data_args, 'use_tuple_infonce', False)
                        and not getattr(LazySupervisedDataset, "_tuple_gap_warned", False)):
                    LazySupervisedDataset._tuple_gap_warned = True
                    rank0_print(
                        f"[TupleInfoNCE] no tuple data for sample {i} "
                        f"(video={video is not None}, audio={audio is not None}, "
                        f"label={bool(label)}). obj1 is 0 for any batch holding it."
                    )
                data_dict["aug_pixel_values_videos"] = None
                data_dict["aug_video_grid_thw"] = None
                data_dict["aug_video_second_per_grid"] = None
                data_dict["aug_input_features"] = None
                data_dict["aug_feature_attention_mask"] = None
                data_dict["aug_input_raw_wav"] = None
                data_dict["tuple_pos_input_ids"] = None
                data_dict["neg_text_input_ids"] = None
                data_dict["audio_only_input_ids"] = None
                data_dict["video_only_input_ids"] = None
                data_dict["ta_input_ids"] = None
                data_dict["tv_input_ids"] = None
                data_dict["av_input_ids"] = None

            if "neg_text" in sources[0]: # and sources[0].get("type", "retrieval") == "give_neg":
                neg_text = sources[0]["neg_text"]
                neg_text = [it + "<|im_end|>" for it in neg_text]
                all_text = [label] + neg_text
                all_label_res = self.tokenizer(all_text, padding=True, padding_side="left", return_tensors="pt")
                all_ids = all_label_res["input_ids"]
                all_attention_mask = all_label_res["attention_mask"]
                data_dict["all_ids"] = all_ids
                data_dict["all_attention_mask"] = all_attention_mask
                # data_dict["all_names"] = all_text
            else:
                data_dict["all_ids"] = None
                data_dict["all_attention_mask"] = None

            if self.data_args.run_test:
                if sources[0].get("type", "retrieval") != "sft":
                    if "neg_text" in sources[0]:
                        neg_text = sources[0]["neg_text"]
                        neg_text = [it + "<|im_end|>" for it in neg_text]
                        all_text = [label] + neg_text
                        all_label_res = self.tokenizer(all_text, padding=True, padding_side="left", return_tensors="pt")
                        all_ids = all_label_res["input_ids"]
                        all_attention_mask = all_label_res["attention_mask"]
                        data_dict["all_ids"] = all_ids
                        data_dict["all_attention_mask"] = all_attention_mask
                        data_dict["all_names"] = all_text
                
                if "sft" in sources[0].get("type", "retrieval"):
                    labels = data_dict.pop("labels", None)
                    len_input = sum(labels[0] == IGNORE_INDEX)
                    data_dict["input_ids"] = data_dict["input_ids"][:, :len_input]
                    data_dict["attention_mask"] = torch.ones_like(data_dict["input_ids"])
                    labels = None

                if "video" in sources[0]:
                    data_dict["video"] = sources[0]["video"]
                elif "frame_dir" in sources[0]:
                    data_dict["video"] = sources[0]["frame_dir"]
                else:
                    data_dict["video"] = None

                data_dict["image"] = sources[0].get("image", None)

                if "audio" in sources[0]:
                    data_dict["audio"] = sources[0]["audio"]
                else:
                    data_dict["audio"] = None

                data_dict["prompt"] = sources[0]["conversations"][:-1]
                data_dict["ref"] = sources[0]["conversations"][-1]["value"]

            return data_dict

        except Exception as e:
            print(f"Error: {e}, line: {e.__traceback__.tb_lineno}")
            # raise e
            if self.data_args.run_test:
                print(f"Error loading {sources[0]}")
                return None
            else:
                randidx = random.choice(self.type_dict[sources[0].get("type", "retrieval")])
                return self.__getitem__(randidx)

@dataclass
class DataCollatorForOmniDataset(object):
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        assert all([it["type"] == instances[0]["type"] for it in instances])

        if instances[0]["type"] != "sft":
            input_ids, label_ids = tuple([instance[key] for instance in instances] for key in ("input_ids", "label_ids"))
            input_ids = [ids.squeeze(0) for ids in input_ids]
            input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=PAD_TOKEN_ID, padding_side='left')
            attention_mask = input_ids.ne(PAD_TOKEN_ID).to(torch.int64)

            if label_ids[0] is not None:
                label_ids = [ids.squeeze(0) for ids in label_ids]
                label_ids = torch.nn.utils.rnn.pad_sequence(label_ids, batch_first=True, padding_value=PAD_TOKEN_ID, padding_side='left')
                label_attention_mask = label_ids.ne(PAD_TOKEN_ID).to(torch.int64)
            else:
                label_ids, label_attention_mask = None, None

            if instances[0]["labels"] is not None:
                labels = [instance["labels"] for instance in instances]
                labels = [ids.squeeze(0) for ids in labels]
                labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX, padding_side='left')
            else:
                labels = None
        else:
            input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
            input_ids = [ids.squeeze(0) for ids in input_ids]
            labels = [ids.squeeze(0) for ids in labels]
            input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=PAD_TOKEN_ID, padding_side='right')
            labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX, padding_side='right')
            attention_mask = input_ids.ne(PAD_TOKEN_ID).to(torch.int64)
            label_ids, label_attention_mask = None, None

        images = [instance["pixel_values"] for instance in instances if "pixel_values" in instance and instance["pixel_values"] is not None]
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            concat_grid_thw = torch.cat([instance["image_grid_thw"] for instance in instances if "image_grid_thw" in instance and instance["image_grid_thw"] is not None], dim=0)
        else:
            concat_images = None
            concat_grid_thw = None

        videos = [instance["pixel_values_videos"] for instance in instances if "pixel_values_videos" in instance and instance["pixel_values_videos"] is not None]
        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            concat_video_grid_thw = torch.cat([instance["video_grid_thw"] for instance in instances if "video_grid_thw" in instance and instance["video_grid_thw"] is not None], dim=0)
            video_second_per_grid = torch.tensor([instance["video_second_per_grid"] for instance in instances if "video_second_per_grid" in instance and instance["video_second_per_grid"] is not None], dtype=torch.float32)
        else:
            concat_videos = None
            concat_video_grid_thw = None
            video_second_per_grid = None

        audios = [instance["input_features"] for instance in instances if "input_features" in instance and instance["input_features"] is not None]
        if len(audios) != 0:
            concat_audios = torch.cat([audio for audio in audios], dim=0)
            concat_feature_attention_mask = torch.cat([instance["feature_attention_mask"] for instance in instances if "feature_attention_mask" in instance and instance["feature_attention_mask"] is not None], dim=0)
        else:
            concat_audios = None
            concat_feature_attention_mask = None

        if any(instance["input_raw_wav"] is not None for instance in instances):
            input_raw_wav = [instance["input_raw_wav"] for instance in instances]
        else:
            input_raw_wav = None

        types = [instance["type"] for instance in instances]
        if any([it["all_ids"] is not None for it in instances]):
            all_ids = [instance["all_ids"] for instance in instances]
            all_attention_mask = [instance["all_attention_mask"] for instance in instances]
        else:
            all_ids = None
            all_attention_mask = None

        if instances[0]["pos_input_ids"] is not None:
            pos_input_ids = [instance["pos_input_ids"] for instance in instances]
            pos_input_ids = [ids.squeeze(0) for ids in pos_input_ids]
            pos_input_ids = torch.nn.utils.rnn.pad_sequence(pos_input_ids, batch_first=True, padding_value=PAD_TOKEN_ID, padding_side='left')
            pos_attention_mask = pos_input_ids.ne(PAD_TOKEN_ID).to(torch.int64)
        else:
            pos_input_ids = None
            pos_attention_mask = None

        pos_images = [instance["pos_pixel_values"] for instance in instances]
        if pos_images[0] is not None:
            concat_pos_images = torch.cat([image for image in pos_images], dim=0)
            concat_pos_grid_thw = torch.cat([instance["pos_image_grid_thw"] for instance in instances], dim=0)
        else:
            concat_pos_images = None
            concat_pos_grid_thw = None

        pos_videos = [instance["pos_pixel_values_videos"] for instance in instances]
        if pos_videos[0] is not None:
            concat_pos_videos = torch.cat([video for video in pos_videos], dim=0)
            concat_pos_video_grid_thw = torch.cat([instance["pos_video_grid_thw"] for instance in instances], dim=0)
            pos_video_second_per_grid = torch.tensor([instance["pos_video_second_per_grid"] for instance in instances], dtype=torch.float32)
        else:
            concat_pos_videos = None
            concat_pos_video_grid_thw = None
            pos_video_second_per_grid = None


        # TupleInfoNCE: augmented video (all samples must have it, otherwise skip)
        aug_videos = [instance.get("aug_pixel_values_videos") for instance in instances]
        if all(v is not None for v in aug_videos):
            concat_aug_videos = torch.cat([v for v in aug_videos], dim=0)
            concat_aug_video_grid_thw = torch.cat([instance["aug_video_grid_thw"] for instance in instances], dim=0)
            aug_video_second_per_grid = torch.tensor([instance["aug_video_second_per_grid"] for instance in instances], dtype=torch.float32)
        else:
            concat_aug_videos = None
            concat_aug_video_grid_thw = None
            aug_video_second_per_grid = None

        # TupleInfoNCE: augmented audio (all samples must have it, otherwise skip)
        aug_audios = [instance.get("aug_input_features") for instance in instances]
        if all(a is not None for a in aug_audios):
            concat_aug_audios = torch.cat([a for a in aug_audios], dim=0)
            concat_aug_feature_attention_mask = torch.cat([instance["aug_feature_attention_mask"] for instance in instances], dim=0)
        else:
            concat_aug_audios = None
            concat_aug_feature_attention_mask = None

        if all(instance.get("aug_input_raw_wav") is not None for instance in instances):
            aug_input_raw_wav = [instance.get("aug_input_raw_wav") for instance in instances]
        else:
            aug_input_raw_wav = None

        # TupleInfoNCE: sub-modal input_ids (including tuple_pos_input_ids for positive pair)
        submodal_keys = ['tuple_pos_input_ids', 'neg_text_input_ids', 'audio_only_input_ids', 'video_only_input_ids', 'ta_input_ids', 'tv_input_ids', 'av_input_ids']
        submodal_batch = {}
        for key in submodal_keys:
            vals = [instance.get(key) for instance in instances]
            if all(v is not None for v in vals):
                vals = [v.squeeze(0) for v in vals]
                padded = torch.nn.utils.rnn.pad_sequence(vals, batch_first=True, padding_value=PAD_TOKEN_ID, padding_side='left')
                submodal_batch[key] = padded
            else:
                submodal_batch[key] = None

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": concat_images,
            "image_grid_thw": concat_grid_thw,
            "pixel_values_videos": concat_videos,
            "video_grid_thw": concat_video_grid_thw,
            "video_second_per_grid": video_second_per_grid,
            "input_features": concat_audios,
            "feature_attention_mask": concat_feature_attention_mask,
            "label_ids": label_ids,
            "label_attention_mask": label_attention_mask,
            "use_audio_in_video": concat_video_grid_thw is not None and concat_audios is not None,
            "types": types,
            "all_ids": all_ids,
            "all_attention_mask": all_attention_mask,
            "labels": labels,
            "pos_pixel_values":  concat_pos_images,
            "pos_image_grid_thw": concat_pos_grid_thw,
            "pos_input_ids": pos_input_ids,
            "pos_attention_mask": pos_attention_mask,
            "pos_pixel_values_videos": concat_pos_videos,
            "pos_video_grid_thw": concat_pos_video_grid_thw,
            "pos_video_second_per_grid": pos_video_second_per_grid,
            "input_raw_wav": input_raw_wav,
            # TupleInfoNCE fields
            "aug_pixel_values_videos": concat_aug_videos,
            "aug_video_grid_thw": concat_aug_video_grid_thw,
            "aug_video_second_per_grid": aug_video_second_per_grid,
            "aug_input_features": concat_aug_audios,
            "aug_feature_attention_mask": concat_aug_feature_attention_mask,
            "aug_input_raw_wav": aug_input_raw_wav,
            **submodal_batch,
        }
        if any(instance.get("text") is not None for instance in instances):
            batch["texts"] = [instance.get("text") for instance in instances]
            batch["ids"] = [instance.get("id") for instance in instances]

        return batch

def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_args=data_args)
    data_collator = DataCollatorForOmniDataset()
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


