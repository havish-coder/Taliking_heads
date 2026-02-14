import os
import cv2
import numpy as np
import warnings
from pathlib import Path
from tqdm import tqdm
from moviepy import VideoFileClip, AudioFileClip
from PIL import Image
import torch
from rtmlib import Wholebody

# ==========================================
#      CONFIGURATION
# ==========================================
SCRIPT_DIR = Path(__file__).parent.resolve()
RAW_VIDEO_DIR = SCRIPT_DIR / "talkvid_sample"
OUTPUT_ROOT = SCRIPT_DIR / "processed_dataset_wholebody"

PROCESSED_VIDEOS_DIR = OUTPUT_ROOT / "videos_24fps"
POSE_OUTPUT_DIR = OUTPUT_ROOT / "pose_data"
FINAL_VIDEO_DIR = OUTPUT_ROOT / "final_videos"
AUDIO_OUTPUT_DIR = OUTPUT_ROOT / "audio"
MODEL_DIR = SCRIPT_DIR / "dwpose_models"

TARGET_SIZE = 768
TARGET_FPS = 24

warnings.filterwarnings("ignore")

# ==========================================
#      MODEL LOADING
# ==========================================
print("🚀 Loading advanced DWPose-l 384x288 model with OpenCV backend...")
det_model_path = str(MODEL_DIR / "yolox_l.onnx")
pose_model_path = str(MODEL_DIR / "dw-ll_ucoco_384.onnx")

wholebody = Wholebody(
    det=det_model_path,
    pose=pose_model_path,
    to_openpose=False,
    backend='opencv',
    device='cuda' if torch.cuda.is_available() else 'cpu'
)
print("✅ Model loaded successfully")

# ==========================================
#      UTILITIES
# ==========================================
def convert_fps(src_path, tgt_path, tgt_fps=TARGET_FPS):
    if tgt_path.exists():
        return True
    try:
        clip = VideoFileClip(str(src_path))
        tgt_path.parent.mkdir(parents=True, exist_ok=True)
        clip.write_videofile(
            str(tgt_path),
            fps=tgt_fps,
            codec='libx264',
            audio_codec='aac',
            logger=None,
            threads=4
        )
        clip.close()
        return True
    except Exception as e:
        print(f"FPS conversion failed: {e}")
        return False

def resize_and_pad_whole(img, target_size):
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h))
    top = (target_size - new_h) // 2
    bottom = target_size - new_h - top
    left = (target_size - new_w) // 2
    right = target_size - new_w - left
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right,
                                cv2.BORDER_CONSTANT, value=(0, 0, 0))
    pad_params = (top, left, new_h, new_w)
    return padded, pad_params

def save_audio(video_path, save_dir):
    save_dir.mkdir(parents=True, exist_ok=True)
    audio_path = save_dir / f"{video_path.stem}.wav"
    if audio_path.exists():
        return
    try:
        clip = AudioFileClip(str(video_path))
        clip.write_audiofile(str(audio_path), logger=None)
        clip.close()
    except Exception as e:
        print(f"Audio extraction failed: {e}")

def map_raw_to_padded(coords_raw, pad_params, orig_h, orig_w):
    """Map raw pixel coordinates from original frame to padded output image."""
    top, left, new_h, new_w = pad_params
    scale_w = new_w / orig_w
    scale_h = new_h / orig_h
    x_scaled = coords_raw[:, 0] * scale_w
    y_scaled = coords_raw[:, 1] * scale_h
    x_padded = x_scaled + left
    y_padded = y_scaled + top
    return np.stack([x_padded, y_padded], axis=1)

# ==========================================
#      POSE PARSING (raw coordinates)
# ==========================================
def parse_rtmlib_output(keypoints, scores, pad_params, orig_h, orig_w):
    """
    keypoints: raw pixel coordinates (num_persons, 133, 2)
    scores: confidence scores (num_persons, 133)
    """
    if len(keypoints) == 0:
        return None, None, None

    # Take first person
    kps = keypoints[0]          # (133, 2) raw pixels
    conf = scores[0]             # (133,)

    # Body (first 17 keypoints)
    body_xy = map_raw_to_padded(kps[:17, :2], pad_params, orig_h, orig_w)
    body_conf = conf[:17].reshape(-1, 1)
    body_kps = np.hstack([body_xy, body_conf])
    bodies = {'candidate': body_kps, 'score': body_conf.flatten()}

    # Face (indices 23-90 = 68 points)
    if kps.shape[0] >= 91:
        face_xy = map_raw_to_padded(kps[23:91, :2], pad_params, orig_h, orig_w)
        faces = [face_xy]
    else:
        faces = []

    # Hands (left: 91-112, right: 112-133)
    if kps.shape[0] >= 133:
        left_hand_xy = map_raw_to_padded(kps[91:112, :2], pad_params, orig_h, orig_w)
        right_hand_xy = map_raw_to_padded(kps[112:133, :2], pad_params, orig_h, orig_w)
        hands = np.stack([left_hand_xy, right_hand_xy], axis=0)   # (2,21,2)
    else:
        hands = np.array([])

    return bodies, faces, hands

# ==========================================
#      MAIN PROCESSING LOOP
# ==========================================
if __name__ == "__main__":
    print(f"Working directory: {SCRIPT_DIR}")
    print(f"Output will be saved to: {OUTPUT_ROOT}")

    video_files = list(RAW_VIDEO_DIR.glob("*.[mM][pP]4"))
    if not video_files:
        print(f"No MP4 files found in {RAW_VIDEO_DIR}")
        exit()

    for video_file in tqdm(video_files, desc="Processing videos"):
        std_path = PROCESSED_VIDEOS_DIR / f"{video_file.stem}_24fps.mp4"
        final_path = FINAL_VIDEO_DIR / f"{video_file.stem}.mp4"
        pose_dir = POSE_OUTPUT_DIR / video_file.stem

        FINAL_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        pose_dir.mkdir(parents=True, exist_ok=True)

        if not convert_fps(video_file, std_path):
            print(f"Skipping {video_file.name}: FPS conversion failed")
            continue
        save_audio(std_path, AUDIO_OUTPUT_DIR)

        # Read video frames
        cap = cv2.VideoCapture(str(std_path))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        if len(frames) == 0:
            print(f"No frames in {video_file.name}")
            continue

        orig_h, orig_w = frames[0].shape[:2]

        # Prepare video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(final_path), fourcc, TARGET_FPS, (TARGET_SIZE, TARGET_SIZE))

        for i, frame in enumerate(tqdm(frames, desc=f"  {video_file.stem}", leave=False)):
            final_frame, pad_params = resize_and_pad_whole(frame, TARGET_SIZE)
            out.write(cv2.cvtColor(final_frame, cv2.COLOR_RGB2BGR))

            # Run pose estimation (rtmlib expects BGR)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            keypoints, scores = wholebody(frame_bgr)   # returns raw pixel coordinates

            bodies, faces, hands = parse_rtmlib_output(keypoints, scores, pad_params, orig_h, orig_w)

            pose_data = {
                'num': i,
                'pad_params': pad_params,
                'orig_size': (orig_h, orig_w),
                'bodies': bodies if bodies is not None else {'candidate': np.array([]), 'score': np.array([])},
                'faces': faces if faces is not None else [],
                'hands': hands if hands is not None else np.array([]),
            }
            np.save(str(pose_dir / f"{i:06d}.npy"), pose_data)

        out.release()
        print(f"✅ Saved: {final_path.name}")

    print("\n🎉 All done! Whole-body videos + pose keypoints saved.")