import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
import pickle
import pyttsx3
import threading
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


model = load_model("asl_lstm_model.keras")

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

labels = label_encoder.classes_


SEQUENCE_LENGTH = 20   
sequence = []

STABLE_THRESHOLD = 8
CONFIDENCE_THRESHOLD = 0.80
SPACE_COOLDOWN = 15

current_word = ""
last_letter = ""
stable_count = 0
space_cooldown_counter = 0


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


root = tk.Tk()
root.title("ASL Gesture to Speech")
root.geometry("950x700")
root.configure(bg="#121212")

style = ttk.Style(root)
style.theme_use("clam")
style.configure(
    "TButton",
    background="#333333",
    foreground="white",
    font=("Segoe UI", 12),
    padding=10
)

video_label = tk.Label(root, bg="#121212")
video_label.pack(pady=10)

word_var = tk.StringVar(value="")
confidence_var = tk.StringVar(value="Confidence: --")

tk.Label(
    root,
    textvariable=word_var,
    font=("Segoe UI", 30, "bold"),
    fg="#4FC3F7",
    bg="#121212"
).pack(pady=10)

tk.Label(
    root,
    textvariable=confidence_var,
    font=("Segoe UI", 14),
    fg="#81C784",
    bg="#121212"
).pack()


engine = pyttsx3.init()
voices = engine.getProperty("voices")
voice_var = tk.StringVar(value=voices[0].name)

ttk.Label(root, text="Voice:", background="#121212", foreground="white").pack(pady=(10, 0))

voice_menu = ttk.Combobox(
    root,
    textvariable=voice_var,
    values=[v.name for v in voices],
    state="readonly",
    width=40
)
voice_menu.pack(pady=5)

def speak_text():
    text = word_var.get().strip()
    if not text:
        return

    def run():
        e = pyttsx3.init()
        e.setProperty("rate", 150)
        for v in e.getProperty("voices"):
            if v.name == voice_var.get():
                e.setProperty("voice", v.id)
                break
        e.say(text)
        e.runAndWait()
        e.stop()

    threading.Thread(target=run, daemon=True).start()

btn_frame = tk.Frame(root, bg="#121212")
btn_frame.pack(pady=15)

speak_btn = ttk.Button(btn_frame, text="🔊 Speak", command=speak_text)
speak_btn.grid(row=0, column=0, padx=15)
speak_btn.config(state=tk.DISABLED)

def clear_word():
    global current_word, sequence
    current_word = ""
    sequence = []
    word_var.set("")
    speak_btn.config(state=tk.DISABLED)

clear_btn = ttk.Button(btn_frame, text="🧹 Clear", command=clear_word)
clear_btn.grid(row=0, column=1, padx=15)


def update_frame():
    global current_word, last_letter, stable_count, space_cooldown_counter, sequence

    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = hand_landmarker.detect(mp_image)

    predicted_letter = None
    confidence = 0.0

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]

        
        for lm in hand:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

        
        wrist_x, wrist_y = hand[0].x, hand[0].y
        landmarks = []

        for lm in hand:
            landmarks.extend([
                lm.x - wrist_x,
                lm.y - wrist_y
            ])

        if len(landmarks) == 42:
            sequence.append(landmarks)
            sequence = sequence[-SEQUENCE_LENGTH:]

            if len(sequence) == SEQUENCE_LENGTH:
                X = np.array(sequence).reshape(1, SEQUENCE_LENGTH, 42)
                probs = model.predict(X, verbose=0)[0]
                idx = np.argmax(probs)
                predicted_letter = labels[idx]
                confidence = probs[idx]

  
    if predicted_letter and confidence >= CONFIDENCE_THRESHOLD:
        display = "␣" if predicted_letter.upper() == "SPACE" else predicted_letter
        confidence_var.set(f"{display} ({confidence:.2f})")

        if predicted_letter == last_letter:
            stable_count += 1
        else:
            stable_count = 0
            last_letter = predicted_letter

        if stable_count >= STABLE_THRESHOLD:
            if predicted_letter.upper() == "SPACE":
                if space_cooldown_counter == 0:
                    current_word += " "
                    space_cooldown_counter = SPACE_COOLDOWN
            else:
                current_word += predicted_letter

            word_var.set(current_word)
            speak_btn.config(state=tk.NORMAL)
            stable_count = 0
    else:
        confidence_var.set("Rejected")
        stable_count = 0
        last_letter = ""

    if space_cooldown_counter > 0:
        space_cooldown_counter -= 1

    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    root.after(10, update_frame)


update_frame()
root.mainloop()

cap.release()
cv2.destroyAllWindows()
hand_landmarker.close()