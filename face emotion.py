import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from deepface import DeepFace
import threading
import numpy as np
import os
from datetime import datetime

class EmotionRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time Face Emotion Recognition")
        self.root.geometry("1000x700")

        # Video capture
        self.cap = None
        self.running = False
        self.emotion_history = []
        self.log_file = "emotions_log.txt"

        # GUI elements
        self.start_button = ttk.Button(root, text="Start", command=self.start_recognition)
        self.start_button.pack(pady=10)

        self.stop_button = ttk.Button(root, text="Stop", command=self.stop_recognition, state=tk.DISABLED)
        self.stop_button.pack(pady=10)

        self.label = tk.Label(root, text="Click Start to begin emotion recognition.")
        self.label.pack(pady=10)

        # Current emotion label
        self.emotion_label = tk.Label(root, text="Current Emotion: None", font=("Arial", 14))
        self.emotion_label.pack(pady=10)

        # Frame for canvas and listbox
        self.frame = tk.Frame(root)
        self.frame.pack()

        # Canvas for video display
        self.canvas = tk.Canvas(self.frame, width=640, height=480, bg="black")
        self.canvas.pack(side=tk.LEFT)

        # Listbox for emotion history
        self.history_label = tk.Label(self.frame, text="Recent Emotions:")
        self.history_label.pack(side=tk.TOP)
        self.history_listbox = tk.Listbox(self.frame, height=10, width=30)
        self.history_listbox.pack(side=tk.TOP, padx=10)

    def start_recognition(self):
        self.running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.label.config(text="Running... Look at the camera!")

        # Start video capture in a separate thread to avoid freezing GUI
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not access webcam.")
            self.label.config(text="Error: Could not access webcam.")
            self.running = False
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            return

        threading.Thread(target=self.process_video, daemon=True).start()

    def stop_recognition(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.label.config(text="Stopped. Click Start to resume.")
        self.canvas.delete("all")  # Clear canvas

    def process_video(self):
        # Use local haarcascade file
        cascade_path = 'haarcascade_frontalface_default.xml'
        if not os.path.exists(cascade_path):
            messagebox.showerror("Error", f"Haar cascade file not found: {cascade_path}")
            return
        face_cascade = cv2.CascadeClassifier(cascade_path)

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                break

            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            current_emotion = "None"
            for (x, y, w, h) in faces:
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

                # Extract face ROI
                face_roi = frame[y:y+h, x:x+w]

                try:
                    # Analyze emotion using DeepFace
                    result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
                    emotion = result[0]['dominant_emotion']
                    confidence = max(result[0]['emotion'].values())
                    current_emotion = f"{emotion} ({confidence:.2f})"

                    # Display emotion on frame
                    cv2.putText(frame, current_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                    # Update history and log
                    self.update_emotion_history(emotion, confidence)
                    self.log_emotion(emotion, confidence)
                except Exception as e:
                    cv2.putText(frame, "Analyzing...", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Update current emotion label
            self.emotion_label.config(text=f"Current Emotion: {current_emotion}")

            # Convert frame to RGB for Tkinter canvas
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(frame_rgb, (640, 480))
            img_tk = tk.PhotoImage(data=cv2.imencode('.ppm', img)[1].tobytes())

            # Update canvas
            self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            self.canvas.image = img_tk  # Keep reference

        self.cap.release()

    def update_emotion_history(self, emotion, confidence):
        timestamp = datetime.now().strftime("%H:%M:%S")
        entry = f"{timestamp}: {emotion} ({confidence:.2f})"
        self.emotion_history.append(entry)
        if len(self.emotion_history) > 5:
            self.emotion_history.pop(0)
        self.history_listbox.delete(0, tk.END)
        for item in self.emotion_history:
            self.history_listbox.insert(tk.END, item)

    def log_emotion(self, emotion, confidence):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.log_file, "a") as f:
            f.write(f"{timestamp}: {emotion} ({confidence:.2f})\n")

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionRecognitionApp(root)
    root.mainloop()
