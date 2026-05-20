import cv2
import pandas as pd
import os
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp


MODEL_PATH = "hand_landmarker.task"

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)

hand_landmarker = vision.HandLandmarker.create_from_options(options)


cap = cv2.VideoCapture(0)

data = []
labels = []

current_label = 'Sohini'
sample_limit = 300

print(f"[INFO] Collecting data for sign: {current_label}")


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    result = hand_landmarker.detect(mp_image)

    if result.hand_landmarks:
        hand_landmarks = result.hand_landmarks[0]

        
        for lm in hand_landmarks:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

        
        wrist_x = hand_landmarks[0].x
        wrist_y = hand_landmarks[0].y

        landmarks = []
        for lm in hand_landmarks:
            landmarks.extend([
                lm.x - wrist_x,
                lm.y - wrist_y
            ])

        if len(landmarks) == 42:
            data.append(landmarks)
            labels.append(current_label)

    cv2.putText(
        frame,
        f"Collected: {len(labels)} / {sample_limit}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow("ASL Data Collection (MediaPipe Tasks)", frame)

    if len(labels) >= sample_limit:
        print(f"[INFO] Collected {sample_limit} samples for {current_label}")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
hand_landmarker.close()


df = pd.DataFrame(data)
df['label'] = labels

csv_file = "asl_data.csv"

if os.path.exists(csv_file):
    df.to_csv(csv_file, mode='a', index=False, header=False)
else:
    df.to_csv(csv_file, index=False)

print("[INFO] Data saved to asl_data.csv")