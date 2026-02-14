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
POSE_FILE = BASE / "pose_data" / VIDEO_NAME / f"{FRAME_INDEX:06d}.npy"
OUTPUT_IMAGE = Path("verification.jpg")

# =============================
def main():

    # ---- Load video frame ----
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

    # ---- Load pose file ----
    if not POSE_FILE.exists():
        print("❌ Pose file not found:", POSE_FILE)
        return

    data = np.load(POSE_FILE, allow_pickle=True)

    # Case 1: dictionary
    if isinstance(data.item() if data.size == 1 else None, dict):
        data = data.item()

        bodies = data.get("bodies", {})
        if bodies.get("candidate", np.array([])).size > 0:
            for x, y, conf in bodies["candidate"]:
                if conf > 0.3:
                    cv2.circle(frame, (int(x), int(y)), 4, (0,255,0), -1)

        faces = data.get("faces", [])
        if faces:
            for pt in faces[0]:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, (255,0,0), -1)

        hands = data.get("hands", np.array([]))
        if hands.size > 0:
            for hand in hands:
                for pt in hand:
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, (0,255,255), -1)

    # Case 2: raw numpy array
    elif isinstance(data, np.ndarray):
        if data.ndim == 2 and data.shape[1] >= 2:
            for pt in data:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 3, (0,255,0), -1)
        else:
            print("⚠ Unknown pose format:", data.shape)

    else:
        print("⚠ Unsupported pose format")
        return

    # ---- Save result ----
    cv2.imwrite(str(OUTPUT_IMAGE), frame)
    print("✅ Verification image saved:", OUTPUT_IMAGE)

    cv2.imshow("Verification", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
