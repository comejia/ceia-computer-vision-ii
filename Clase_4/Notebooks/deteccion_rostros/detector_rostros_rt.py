

import cv2
import os

face_cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
eye_cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_eye.xml"))
smile_cascade = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, "haarcascade_smile.xml"))

if face_cascade.empty() or eye_cascade.empty() or smile_cascade.empty():
    raise RuntimeError("No se pudieron cargar los clasificadores Haar de OpenCV.")

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la cámara.")

def draw_box(img, x, y, w, h, color=(255, 0, 0), thickness=2):
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=30, minSize=(120, 120))

    status_msg = []
    for (fx, fy, fw, fh) in faces:
        draw_box(frame, fx, fy, fw, fh, color=(255, 0, 0), thickness=2)

        roi_gray = gray[fy:fy + fh, fx:fx + fw]
        roi_color = frame[fy:fy + fh, fx:fx + fw]

        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=15, minSize=(25, 25))
        eyes_sorted = sorted(eyes, key=lambda e: e[0])
        for (ex, ey, ew, eh) in eyes:
            draw_box(roi_color, ex, ey, ew, eh, color=(0, 255, 0), thickness=2)

        wink_detected = False
        if len(eyes_sorted) == 1:
            wink_detected = True
            ex, ey, ew, eh = eyes_sorted[0]
            cx_eye = fx + ex + ew // 2
            mid_face = fx + fw // 2
            if cx_eye < mid_face:
                status_msg.append("Ojo derecho cerrado")
            else:
                status_msg.append("Ojo izquierdo cerrado")

        smiles = smile_cascade.detectMultiScale(
            roi_gray, scaleFactor=1.1, minNeighbors=30, minSize=(40, 40)
        )

        mouth_open = False
        for (sx, sy, sw, sh) in smiles:
            draw_box(roi_color, sx, sy, sw, sh, color=(255, 0, 255), thickness=2)
            if sw / float(fw) > 0.25 and sh / float(fh) > 0.10:
                mouth_open = True

        if mouth_open:
            status_msg.append("")
        else:
            status_msg.append("")

        if not wink_detected and len(eyes_sorted) >= 2:
            status_msg.append("Ambos ojos abiertos")

    y0 = 30
    for msg in status_msg[:3]:
        cv2.putText(frame, msg, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 220, 50), 2, cv2.LINE_AA)
        y0 += 30

    cv2.putText(frame, "ESC para salir", (10, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.imshow("Deteccion de rostro, ojos y boca", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
