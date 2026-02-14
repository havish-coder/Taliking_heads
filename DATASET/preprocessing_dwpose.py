import os
import cv2
import numpy as np
import warnings
from pathlib import Path
from tqdm import tqdm
from moviepy import VideoFileClip
import torch
from rtmlib import Wholebody

# ==========================================
# CONFIGURATION
# ==========================================
SCRIPT_DIR = Path(__file__).parent.resolve()
RAW_VIDEO_DIR = SCRIPT_DIR / "Video_WithoutAudio"
OUTPUT_ROOT = SCRIPT_DIR / "processed_dataset_wholebody"

PROCESSED_VIDEOS_DIR = OUTPUT_ROOT / "videos_24fps"
POSE_OUTPUT_DIR = OUTPUT_ROOT / "pose_data_single"
FINAL_VIDEO_DIR = OUTPUT_ROOT / "final_videos"
MODEL_DIR = SCRIPT_DIR / "dwpose_models"

TARGET_SIZE = 768
TARGET_FPS = 24

warnings.filterwarnings("ignore")

# ==========================================
# MODEL LOADING
# ==========================================
print("🚀 Loading model...")
det_model_path = str(MODEL_DIR / "yolox_l.onnx")
pose_model_path = str(MODEL_DIR / "dw-ll_ucoco_384.onnx")

wholebody = Wholebody(
    det=det_model_path,
    pose=pose_model_path,
    to_openpose=False,
    backend='opencv',
    device='cuda' if torch.cuda.is_available() else 'cpu'
)
print("✅ Model loaded")

# ==========================================
# FPS CONVERSION
# ==========================================
def convert_fps(src_path, tgt_path):
    if tgt_path.exists():
        return True
    try:
        clip = VideoFileClip(str(src_path))
        tgt_path.parent.mkdir(parents=True, exist_ok=True)
        clip.write_videofile(
            str(tgt_path),
            fps=TARGET_FPS,
            codec='libx264',
            audio=False,
            logger=None
        )
        clip.close()
        return True
    except Exception as e:
        print(f"FPS conversion failed: {e}")
        return False


# ==========================================
# RESIZE + PAD (RETURN PARAMS)
# ==========================================
def resize_and_pad_whole(img):
    orig_h, orig_w = img.shape[:2]

    scale = TARGET_SIZE / max(orig_h, orig_w)
    new_h, new_w = int(orig_h * scale), int(orig_w * scale)

    resized = cv2.resize(img, (new_w, new_h))

    top = (TARGET_SIZE - new_h) // 2
    bottom = TARGET_SIZE - new_h - top
    left = (TARGET_SIZE - new_w) // 2
    right = TARGET_SIZE - new_w - left

    padded = cv2.copyMakeBorder(
        resized,
        top, bottom, left, right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )

    pad_params = (top, left, new_h, new_w, orig_h, orig_w)

    return padded, pad_params


# ==========================================
# MAP KEYPOINTS TO PADDED SPACE
# ==========================================
def map_keypoints_to_padded(keypoints, pad_params):
    top, left, new_h, new_w, orig_h, orig_w = pad_params

    if len(keypoints) == 0:
        return np.array([])

    kps = keypoints[0].copy()  # (133,2)

    scale_w = new_w / orig_w
    scale_h = new_h / orig_h

    kps[:, 0] = kps[:, 0] * scale_w + left
    kps[:, 1] = kps[:, 1] * scale_h + top

    return kps


# ==========================================
# MAIN (ONE VIDEO ONLY)
# ==========================================
if __name__ == "__main__":

    video_files = list(RAW_VIDEO_DIR.glob("*.[mM][pP]4"))
    if not video_files:
        print("No MP4 files found.")
        exit()

    video_file = video_files[0]  # 🔥 ONLY FIRST VIDEO

    std_path = PROCESSED_VIDEOS_DIR / f"{video_file.stem}_24fps.mp4"
    final_path = FINAL_VIDEO_DIR / f"{video_file.stem}.mp4"
    pose_output_path = POSE_OUTPUT_DIR / f"{video_file.stem}.npy"

    FINAL_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    POSE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not convert_fps(video_file, std_path):
        exit()

    cap = cv2.VideoCapture(str(std_path))

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if len(frames) == 0:
        print("No frames found.")
        exit()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(final_path), fourcc, TARGET_FPS, (TARGET_SIZE, TARGET_SIZE))

    all_pose_data = []

    for i, frame in enumerate(tqdm(frames)):

        padded_frame, pad_params = resize_and_pad_whole(frame)
        out.write(padded_frame)

        keypoints, scores = wholebody(frame)

        mapped_kps = map_keypoints_to_padded(keypoints, pad_params)

        pose_frame = {
            "frame_index": i,
            "keypoints": mapped_kps,
            "scores": scores[0] if len(scores) > 0 else None
        }

        all_pose_data.append(pose_frame)

    out.release()

    np.save(pose_output_path, all_pose_data)

    print(f"✅ Saved video: {final_path.name}")
    print(f"✅ Saved pose file: {pose_output_path.name}")
    print("🎯 Keypoints now perfectly match final_videos resolution.")


