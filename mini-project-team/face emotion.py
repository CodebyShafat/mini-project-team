import tkinter as tk
from tkinter import *
from matplotlib.pyplot import Image, ImageTk
import cv2
import numpy as np
from keras.models import load_model

# Load Model & Haarcascade
model = load_model("emotion_model.h5")
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

emotion_labels = ['Angry','Disgust','Fear','Happy','Sad','Surprise','Neutral']

# Main Window
root = tk.Tk()
root.title("Face Emotion Recognition - Desktop App")
root.geometry("800x600")
root.configure(bg="#222")

label = Label(root, text="Emotion Recognition", font=("Arial", 20), bg="#222", fg="white")
label.pack()

video_label = Label(root)
video_label.pack()

cap = None

def start_camera():
    global cap
    cap = cv2.VideoCapture(0)
    update_frame()

def stop_camera():
    global cap
    if cap:
        cap.release()
    video_label.config(image="")

def update_frame():
    global cap
    ret, frame = cap.read()
    if not ret:
        return
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48,48))
        roi = roi_gray.astype("float")/255.0
        roi = np.expand_dims(roi, axis=0)
        roi = np.expand_dims(roi, axis=-1)

        preds = model.predict(roi)[0]
        emotion = emotion_labels[np.argmax(preds)]

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,0,0),2)

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    imgtk = ImageTk.PhotoImage(image=img)
    
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)
    video_label.after(10, update_frame)

start_btn = Button(root, text="Start Camera", command=start_camera, font=("Arial", 14), bg="green", fg="white")
start_btn.pack(pady=10)

stop_btn = Button(root, text="Stop Camera", command=stop_camera, font=("Arial", 14), bg="red", fg="white")
stop_btn.pack(pady=10)

root.mainloop()
