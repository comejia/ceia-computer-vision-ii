
import cv2
import numpy as np
import math

from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.python.solutions import pose as mp_pose

def calcular_angulo(a, b, c):
    ax, ay = a
    bx, by = b
    cx, cy = c

    ba = np.array([ax - bx, ay - by])
    bc = np.array([cx - bx, cy - by])

    coseno_angulo = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    coseno_angulo = np.clip(coseno_angulo, -1.0, 1.0)
    angulo = math.degrees(math.acos(coseno_angulo))
    return angulo

def detectar_biceps(landmarks):
    rs = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    re = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    rw = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

    ls = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    le = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    lw = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]

    right_shoulder = np.array([rs.x, rs.y])
    right_elbow = np.array([re.x, re.y])
    right_wrist = np.array([rw.x, rw.y])

    left_shoulder = np.array([ls.x, ls.y])
    left_elbow = np.array([le.x, le.y])
    left_wrist = np.array([lw.x, lw.y])

    angulo_der = calcular_angulo(right_shoulder, right_elbow, right_wrist)
    angulo_izq = calcular_angulo(left_shoulder, left_elbow, left_wrist)

    dist_mano_hombro_der = np.linalg.norm(right_wrist - right_shoulder)
    dist_mano_hombro_izq = np.linalg.norm(left_wrist - left_shoulder)

    umbral_angulo = 80.0
    umbral_dist = 0.35

    gesto_der = angulo_der < umbral_angulo and dist_mano_hombro_der < umbral_dist
    gesto_izq = angulo_izq < umbral_angulo and dist_mano_hombro_izq < umbral_dist

    if gesto_der and gesto_izq:
        return "ambos"
    elif gesto_der:
        return "derecho"
    elif gesto_izq:
        return "izquierdo"
    else:
        return None

cap = cv2.VideoCapture(0)

with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultados = pose.process(frame_rgb)

        if resultados.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                resultados.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
            )

            gesto = detectar_biceps(resultados.pose_landmarks.landmark)
            if gesto is not None:
                texto = f"Gesto de biceps {gesto} detectado"
                cv2.putText(frame, texto, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Deteccion de gesto de biceps", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
