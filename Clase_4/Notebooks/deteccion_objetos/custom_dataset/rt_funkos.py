# -*- coding: utf-8 -*-
"""

@author: Gerardo Vilcamiza
"""

#%%

import time
import cv2
import torch
from ultralytics import YOLO


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Usando dispositivo:", device)

model = YOLO("best.pt")
model.to(device)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la cámara")

prev_time = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    start = time.time()

    results = model.predict(
        source=frame,
        imgsz=640,
        conf=0.7,
        device=device,
        verbose=False
    )

    annotated_frame = results[0].plot()

    end = time.time()
    fps = 1.0 / max(end - start, 1e-6)
    prev_time = fps if prev_time == 0 else 0.9 * prev_time + 0.1 * fps

    cv2.putText(annotated_frame, f"FPS: {prev_time:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("YOLOv11 Real Time", annotated_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or key == 27:
        break

cap.release()
cv2.destroyAllWindows()



