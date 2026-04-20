# -*- coding: utf-8 -*-
"""

@author: Gerardo Vilcamiza
"""

#%%

from ultralytics import YOLO
import cv2

video_path = "oficina.mp4"

model = YOLO("yolo11m.pt")

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("No se pudo abrir el video.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, device=0, conf=0.7, imgsz=1280, verbose=True)

    annotated_frame = results[0].plot()

    cv2.imshow("Detección YOLOv11s", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
