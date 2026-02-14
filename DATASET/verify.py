import cv2
import numpy as np
from pathlib import Path

# =============================
# CHANGE THESE
# =============================
VIDEO_NAME = "000_NBMTc71yrpk"   # without .mp4
FRAME_INDEX = 0

BASE = Path("processed_dataset_wholebody")
VIDEO_PATH = BASE / "final_videos" / f"{VIDEO_NAME}.mp4"
POSE_FILE = BASE / "pose_data_single" / f"{VIDEO_NAME}.npy"

# =============================
def main():

    # ---- Load video frame (PADDED final video) ----
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print("❌ Cannot open video")
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_INDEX)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("❌ Cannot read frame")
        return

    # ---- Load pose file (single file format) ----
    if not POSE_FILE.exists():
        print("❌ Pose file not found:", POSE_FILE)
        return

    data = np.load(POSE_FILE, allow_pickle=True)

    if FRAME_INDEX >= len(data):
        print("❌ Frame index out of range")
        return

    frame_data = data[FRAME_INDEX]

    keypoints = frame_data["keypoints"]
    scores = frame_data["scores"]

    # If multiple persons, take first
    if len(keypoints.shape) == 3:
        keypoints = keypoints[0]
        scores = scores[0]

    print("Drawing", keypoints.shape[0], "points")

    # ---- Draw ALL keypoints ----
    for i in range(len(keypoints)):
        x, y = keypoints[i]
        conf = scores[i]

        if conf > 0.05:  # draw even low confidence
            cv2.circle(frame, (int(x), int(y)), 4, (0, 0, 255), -1)

    # ---- Show result ----
    cv2.imshow("Verification", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()



