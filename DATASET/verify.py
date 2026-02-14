import cv2
import numpy as np
from pathlib import Path

# ==========================================
# CONFIGURATION – change these to match your video
# ==========================================
VIDEO_NAME = "videovideo4ENQEX1y7dM-scene1-scene7"  # without .mp4
FRAME_INDEX = 0                                           # which frame to check

POSE_ROOT = Path("processed_dataset_wholebody/pose_data")
VIDEO_PATH = Path("processed_dataset_wholebody/final_videos") / f"{VIDEO_NAME}.mp4"
OUTPUT_IMAGE = Path("verification.jpg")

# Colors (BGR)
BODY_COLOR = (0, 255, 0)      # green
FACE_COLOR = (255, 0, 0)      # blue
HAND_COLOR = (0, 255, 255)    # yellow
FEET_COLOR = (255, 0, 255)    # magenta

# ==========================================
def main():
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print(f"❌ Cannot open video: {VIDEO_PATH}")
        return
    cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_INDEX)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"❌ Could not read frame {FRAME_INDEX}")
        return

    pose_file = POSE_ROOT / VIDEO_NAME / f"{FRAME_INDEX:06d}.npy"
    if not pose_file.exists():
        print(f"❌ Pose file not found: {pose_file}")
        return

    data = np.load(pose_file, allow_pickle=True).item()

    # Draw body keypoints
    bodies = data.get('bodies', {})
    if bodies.get('candidate', np.array([])).size > 0:
        for x, y, conf in bodies['candidate']:
            if conf > 0.3:
                cv2.circle(frame, (int(x), int(y)), 4, BODY_COLOR, -1)

    # Draw face keypoints
    faces = data.get('faces', [])
    if faces and len(faces) > 0 and faces[0] is not None:
        for pt in faces[0][:, :2]:
            cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, FACE_COLOR, -1)

    # Draw hand keypoints
    hands = data.get('hands', np.array([]))
    if hands.size > 0 and hands.shape[0] >= 2:
        left, right = hands[0], hands[1]
        for pt in left[:, :2]:
            if pt[0] > 0 and pt[1] > 0:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, HAND_COLOR, -1)
        for pt in right[:, :2]:
            if pt[0] > 0 and pt[1] > 0:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 2, HAND_COLOR, -1)

    # Draw feet (if present)
    feet = data.get('feet', np.array([]))
    if feet.size > 0:
        for pt in feet[:, :2]:
            if pt[0] > 0 and pt[1] > 0:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 3, FEET_COLOR, -1)

    cv2.imwrite(str(OUTPUT_IMAGE), frame)
    print(f"✅ Verification image saved to {OUTPUT_IMAGE}")

    cv2.imshow("Verification", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()