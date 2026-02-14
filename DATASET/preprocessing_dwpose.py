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

FINAL_VIDEO_DIR = OUTPUT_ROOT / "final_videos"
POSE_OUTPUT_DIR = OUTPUT_ROOT / "pose_data"
TEMP_24FPS_DIR = OUTPUT_ROOT / "temp_24fps"  # temporary only
MODEL_DIR = SCRIPT_DIR / "dwpose_models"

TARGET_SIZE = 768
TARGET_FPS = 24

warnings.filterwarnings("ignore")

# ==========================================
# MODEL LOADING
# ==========================================
print("Loading DWPose model...")

det_model_path = (MODEL_DIR / "yolox_l.onnx").as_posix()
pose_model_path = (MODEL_DIR / "dw-ll_ucoco_384.onnx").as_posix()

wholebody = Wholebody(
    det=det_model_path,
    pose=pose_model_path,
    to_openpose=False,
    backend='opencv',
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

print("Model loaded successfully.")

# ==========================================
# UTILITIES
# ==========================================
def convert_to_24fps(src_path, tgt_path):
    clip = VideoFileClip(str(src_path))
    tgt_path.parent.mkdir(parents=True, exist_ok=True)

    clip.write_videofile(
        str(tgt_path),
        fps=TARGET_FPS,
        codec='libx264',
        audio=False,
        logger=None,
        threads=4
    )
    clip.close()


def resize_and_pad(img, target_size):
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h))

    top = (target_size - new_h) // 2
    bottom = target_size - new_h - top
    left = (target_size - new_w) // 2
    right = target_size - new_w - left

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )

    return padded, (top, left, new_h, new_w)


def map_coords(coords, pad_params, orig_h, orig_w):
    top, left, new_h, new_w = pad_params
    scale_w = new_w / orig_w
    scale_h = new_h / orig_h

    x_scaled = coords[:, 0] * scale_w
    y_scaled = coords[:, 1] * scale_h

    x_padded = x_scaled + left
    y_padded = y_scaled + top

    return np.stack([x_padded, y_padded], axis=1)


def parse_pose(keypoints, scores, pad_params, orig_h, orig_w):
    if len(keypoints) == 0:
        return None

    kps = keypoints[0]
    conf = scores[0]

    body_xy = map_coords(kps[:17, :2], pad_params, orig_h, orig_w)
    body_conf = conf[:17].reshape(-1, 1)
    body = np.hstack([body_xy, body_conf])

    return body


# ==========================================
# MAIN (ONE SAMPLE ONLY)
# ==========================================
if __name__ == "__main__":

    video_files = list(RAW_VIDEO_DIR.glob("*.[mM][pP]4"))
    if not video_files:
        print("No MP4 files found.")
        exit()

    video_file = video_files[0]
    print(f"\nProcessing sample: {video_file.name}")

    FINAL_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    POSE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TEMP_24FPS_DIR.mkdir(parents=True, exist_ok=True)

    temp_24 = TEMP_24FPS_DIR / f"{video_file.stem}_24fps.mp4"
    final_video_path = FINAL_VIDEO_DIR / video_file.name
    pose_dir = POSE_OUTPUT_DIR / video_file.stem
    pose_dir.mkdir(parents=True, exist_ok=True)

    # 1️⃣ Convert to 24 FPS
    convert_to_24fps(video_file, temp_24)

    # 2️⃣ Read converted video
    cap = cv2.VideoCapture(str(temp_24))
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()

    if len(frames) == 0:
        print("No frames found.")
        exit()

    orig_h, orig_w = frames[0].shape[:2]

    # 3️⃣ Prepare writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        str(final_video_path),
        fourcc,
        TARGET_FPS,
        (TARGET_SIZE, TARGET_SIZE)
    )

    # 4️⃣ Extract pose
    for i, frame in enumerate(tqdm(frames, desc="Extracting pose")):

        padded, pad_params = resize_and_pad(frame, TARGET_SIZE)
        out.write(cv2.cvtColor(padded, cv2.COLOR_RGB2BGR))

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        keypoints, scores = wholebody(frame_bgr)

        pose = parse_pose(keypoints, scores, pad_params, orig_h, orig_w)

        np.save(pose_dir / f"{i:06d}.npy",
                pose if pose is not None else np.array([]))

    out.release()

    # 5️⃣ Delete temporary 24fps file
    if temp_24.exists():
        temp_24.unlink()

    print("\nDone. Only final_videos and pose_data kept.")
