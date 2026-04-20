# -*- coding: utf-8 -*-
"""

@author: gerar
"""

#%%

import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

# ======================================================
# PARÁMETROS GENERALES DEL SISTEMA
# ======================================================

VIDEO_PATH = "autos.mp4"
# Ruta del video de entrada.
# Afecta: si el video tiene distinta resolución/FPS/ángulo de cámara, cambian los valores óptimos de CONF/IOU,
# y también la posición adecuada de LINE_P1/LINE_P2.

MODEL_PATH = "yolo11s.pt"
# Ruta del modelo YOLOv11 a usar.
# Afecta:
# - Precisión: modelos más grandes suelen detectar mejor (menos falsos positivos/negativos).
# - Velocidad: modelos más grandes son más lentos.
# - Clases disponibles: depende del dataset con el que fue entrenado.

CONF = 0.4
# Umbral mínimo de confianza para aceptar una detección.
# Afecta:
# - Si subes CONF: se filtran detecciones débiles -> menos falsos positivos, pero puedes perder autos lejanos/ocultos.
# - Si bajas CONF: aparecen más detecciones -> más recall, pero el tracker recibe más ruido (IDs inestables).
# En conteo: CONF muy alto puede fragmentar el track (mismo auto cambia de ID) y provocar doble conteo.

IOU = 0.6
# Umbral IoU usado por el pipeline de detección/tracking para asociación y/o supresión.
# Afecta:
# - Alto IOU: asociación más estricta -> si la caja cambia mucho entre frames (movimiento, oclusión), se rompe el ID.
# - Bajo IOU: asociación más laxa -> riesgo de confundir autos cercanos y mezclar IDs.
# En conteo: IOU muy alto puede crear IDs nuevos para el mismo auto, y IOU muy bajo puede "fusionar" autos.

TARGET_CLASSES = {"car", "truck", "bus", "motorcycle"}
# Conjunto de clases que SÍ quieres procesar.
# Afecta:
# - Si incluyes clases innecesarias (person, bicycle, etc.), el conteo se contamina.
# - Si omites clases (por ejemplo "truck"), no contarás esos vehículos.

LINE_P1 = (520, 360)
LINE_P2 = (1580, 360)
# Dos puntos que definen la línea de conteo (segmento).
# Afecta:
# - Posición del conteo: si la línea está mal ubicada, contarás antes/después del punto real.
# - Escala: estos puntos están en coordenadas del frame ORIGINAL, antes del resize a 720p.
# Importante: si cambias el tamaño de procesamiento (no solo visualización), tendrías que escalar estos puntos.

TRACKER = "bytetrack.yaml"
# Archivo de configuración del tracker.
# Afecta:
# - Cómo se asocian detecciones a tracks.
# - Cuánto tiempo se mantiene un track sin detección.
# - Umbrales internos (dependen del yaml).
# ByteTrack es robusto para tráfico, pero esos parámetros pueden cambiar muchísimo estabilidad de IDs.

DIST_THRESH = 18
# Distancia máxima permitida para considerar que el centro del bbox está "sobre" el segmento.
# Afecta:
# - Si aumentas DIST_THRESH: más fácil contar cruces, pero suben falsos positivos (cuenta cuando pasa cerca).
# - Si disminuyes DIST_THRESH: cuenta solo cruces muy precisos, pero puede perder cruces reales.

MAX_TRACE_LEN = 40
# Cantidad máxima de puntos de trayectoria guardados por cada ID.
# Afecta:
# - Visual: trayectoria más larga si subes este valor.
# - Rendimiento: más líneas dibujadas por frame.
# - Memoria: se guardan más puntos por vehículo.

CYAN = (255, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
# Colores en BGR (OpenCV usa BGR, no RGB).
# CYAN: se usa como estado "aún no cruzó".
# RED: se usa como estado "ya cruzó" (evento consumado).
# YELLOW: se usa para la trayectoria (señal visual del historial del track).

DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720
# Resolución objetivo SOLO para mostrar el video.
# Afecta:
# - No afecta detección ni tracking (porque se hace después).
# - Mejora visualización en pantalla.
# - Reduce costo de renderizado del imshow (dibujar/mostrar menos pixeles).


# ======================================================
# FUNCIONES GEOMÉTRICAS AUXILIARES
# ======================================================

def side_of_line(p, a, b):
    """
    Determina en qué lado de la recta AB se encuentra el punto P.

    Concepto:
    - Calcula el "producto cruzado" 2D entre el vector AB y AP.
    - El signo del resultado indica el lado:
      positivo: un lado
      negativo: el otro lado
      cercano a cero: muy cerca de la recta

    Por qué sirve:
    - Si en el frame t-1 el punto está a un lado y en el frame t al otro,
      hubo un cruce (cambio de signo).

    Afecta:
    - Si el tracker es inestable (IDs cambian), esta lógica se rompe porque no hay continuidad.
    """
    ax, ay = a
    bx, by = b
    px, py = p
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


def point_on_segment(p, a, b, eps=2.0):
    """
    Verifica si el punto P está suficientemente cerca del segmento AB.

    Concepto:
    - Proyecta P sobre la recta AB.
    - Recorta la proyección al segmento (t entre 0 y 1).
    - Mide la distancia entre P y su proyección.
    - Si esa distancia <= eps, consideramos que P "está sobre" el segmento.

    Por qué sirve:
    - El cambio de signo por sí solo puede ocurrir lejos del segmento
      si el punto está arriba/abajo pero a la izquierda/derecha dependiendo de la orientación.
    - Esta condición actúa como filtro espacial: "cruzó, pero cerca de la línea".

    Afecta:
    - eps (que luego alimentas con DIST_THRESH) controla la sensibilidad.
    """
    p = np.array(p, dtype=np.float32)
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)

    ab = b - a
    ap = p - a
    ab2 = np.dot(ab, ab)

    if ab2 < 1e-6:
        return np.linalg.norm(p - a) <= eps

    t = np.clip(np.dot(ap, ab) / ab2, 0.0, 1.0)
    proj = a + t * ab

    return np.linalg.norm(p - proj) <= eps


def box_center(x1, y1, x2, y2):
    """
    Calcula el centro geométrico del bbox.

    Por qué el centro:
    - Es una representación simple del objeto.
    - Es lo típico para conteo por línea.
    - Evita depender de cambios pequeños en bordes del bbox.

    Afecta:
    - Si el bbox fluctúa mucho (detección inestable), el centro "tiembla" y puede generar cruces falsos.
    - Para tráfico, a veces funciona mejor usar un punto más bajo (por ejemplo el "bottom-center")
      si te interesa cuándo la rueda pasa la línea, no el centro del auto.
    """
    return (int((x1 + x2) / 2), int((y1 + y2) / 2))


# ======================================================
# FUNCIÓN PRINCIPAL
# ======================================================

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    # Abre el video. Si no abre, cap.read() devolverá ret=False.
    # Afecta: si el codec no es compatible o la ruta es incorrecta, no podrás procesar.

    model = YOLO(MODEL_PATH)
    # Carga el modelo en memoria.
    # Afecta: en GPU/CPU según tu entorno; en algunos casos ultralytics usa GPU si está disponible.

    names = model.names
    # Diccionario: id_clase -> nombre_clase (ej: 2 -> "car").
    # Afecta: si el modelo fue entrenado custom, estos nombres cambian.

    prev_side = {}
    # Diccionario: track_id -> último valor de side_of_line.
    # Rol: memoria del "lado anterior" para comparar y detectar cambio de signo (cruce).

    counted = set()
    # Conjunto de track_id ya contados.
    # Rol: garantiza conteo 1 vez por vehículo.
    # Afecta: si el ID cambia por tracking inestable, el mismo auto puede contarse varias veces.

    crossed_total = 0
    # Contador total de cruces detectados.

    traces = defaultdict(list)
    # Diccionario: track_id -> lista de puntos (cx, cy) históricos.
    # Rol: dibujar la trayectoria.
    # Afecta: si MAX_TRACE_LEN es grande y hay muchos vehículos, se dibuja mucho por frame.

    while True:
        ret, frame = cap.read()
        # ret indica si se pudo leer un frame.
        # frame es la imagen original del video en su resolución original.
        if not ret:
            break

        cv2.line(frame, LINE_P1, LINE_P2, CYAN, 2)
        # Dibuja la línea de conteo.
        # Importante: esto se hace en el frame original, antes del resize de visualización.

        results = model.track(
            source=frame,
            conf=CONF,
            iou=IOU,
            tracker=TRACKER,
            persist=True,
            verbose=False
        )
        # model.track hace dos cosas:
        # 1) detección (YOLO) en el frame actual
        # 2) tracking (ByteTrack) asociando detecciones a IDs persistentes
        #
        # Parámetros:
        # - conf: filtra detecciones débiles
        # - iou: afecta asociación y supresión
        # - tracker: define cómo se trackea
        # - persist=True: mantiene estado del tracker entre frames (esencial para IDs)
        # - verbose=False: evita logs en consola

        if results and results[0].boxes is not None:
            boxes = results[0].boxes
            # boxes contiene:
            # - xyxy: coordenadas
            # - cls: clase
            # - id: track_id asignado por el tracker

            xyxy = boxes.xyxy.cpu().numpy()
            # Bounding boxes por detección en formato [x1, y1, x2, y2].
            # Afecta: son coordenadas en el frame original.

            clss = boxes.cls.cpu().numpy().astype(int)
            # IDs de clase por bbox.

            ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else None
            # IDs de tracking. Si el tracker no devuelve IDs, queda None.
            # Afecta: sin IDs no hay conteo robusto, solo detección frame a frame.

            for i in range(len(xyxy)):
                cls_name = names.get(clss[i], "")
                # Convierte el id de clase a nombre.

                if cls_name not in TARGET_CLASSES:
                    continue
                # Filtra solo vehículos.

                x1, y1, x2, y2 = map(int, xyxy[i])
                # Convierte coordenadas a enteros para dibujar en OpenCV.

                track_id = int(ids[i]) if ids is not None else -1
                # ID del objeto.
                # Si es -1 significa que no hay tracking para ese bbox.

                cx, cy = box_center(x1, y1, x2, y2)
                # Centro geométrico del bbox.
                # Es el punto que usaremos para trayectoria y cruce.

                # ----------------------------------------------
                # DIBUJO DE LA TRAYECTORIA (AMARILLO)
                # ----------------------------------------------

                if track_id != -1:
                    traces[track_id].append((cx, cy))
                    # Guarda el punto actual en el historial del ID.

                    if len(traces[track_id]) > MAX_TRACE_LEN:
                        traces[track_id].pop(0)
                    # Limita el historial para que la trayectoria no crezca indefinidamente.

                    for j in range(1, len(traces[track_id])):
                        cv2.line(
                            frame,
                            traces[track_id][j - 1],
                            traces[track_id][j],
                            YELLOW,
                            2
                        )
                    # Dibuja segmentos consecutivos para formar la trayectoria.

                # ----------------------------------------------
                # DETECCIÓN DE CRUCE DE LÍNEA
                # ----------------------------------------------

                if track_id != -1:
                    cur_side = side_of_line((cx, cy), LINE_P1, LINE_P2)
                    # Determina el lado del punto actual respecto a la línea.

                    if track_id not in prev_side:
                        prev_side[track_id] = cur_side
                        # Primer frame en el que vemos este track_id: guardamos su lado inicial.
                    else:
                        prev = prev_side[track_id]
                        prev_side[track_id] = cur_side
                        # Guardamos el lado anterior y lo actualizamos.

                        crossed = (
                            (prev > 0 and cur_side < 0) or
                            (prev < 0 and cur_side > 0)
                        )
                        # Cambio de signo: pasó de un lado al otro.
                        # Ojo: si el punto cae exactamente en la línea, cur_side puede ser 0.

                        if crossed and point_on_segment((cx, cy), LINE_P1, LINE_P2, DIST_THRESH):
                            # Confirmación espacial: el punto está cerca del segmento.
                            # Esto reduce falsos positivos.

                            if track_id not in counted:
                                counted.add(track_id)
                                crossed_total += 1
                                # Conteo 1 vez por ID. Luego el bbox quedará rojo.

                # ----------------------------------------------
                # DIBUJO DEL BOUNDING BOX
                # ----------------------------------------------

                bbox_color = RED if track_id in counted else CYAN
                # Lógica de estado visual:
                # - Si el ID ya cruzó, bbox rojo.
                # - Si no cruzó, bbox cyan.

                cv2.rectangle(frame, (x1, y1), (x2, y2), bbox_color, 2)
                # Dibuja bbox.

                cv2.circle(frame, (cx, cy), 4, bbox_color, -1)
                # Dibuja el centro del bbox, útil para depurar cruce y trayectoria.

                cv2.putText(
                    frame,
                    f"{cls_name} id={track_id}",
                    (x1, max(0, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    bbox_color,
                    2
                )
                # Imprime clase e ID sobre el bbox.

        # ----------------------------------------------
        # REDIMENSIONAMIENTO SOLO PARA VISUALIZACIÓN
        # ----------------------------------------------

        frame_display = cv2.resize(
            frame,
            (DISPLAY_WIDTH, DISPLAY_HEIGHT),
            interpolation=cv2.INTER_AREA
        )
        # Importante:
        # - Esto NO afecta detección ni tracking, porque ya se hizo antes.
        # - Solo mejora visualización y reduce carga de renderizado en pantalla.
        # - La línea y bboxes ya están dibujados sobre el frame original.

        cv2.putText(
            frame_display,
            f"Cruces: {crossed_total}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )
        # Muestra el contador total.

        cv2.imshow("YOLOv11 Tracking + Line Crossing", frame_display)
        # Ventana de visualización.

        if cv2.waitKey(1) & 0xFF in [27, ord("q")]:
            break
        # Permite salir con ESC o con la letra q.

    cap.release()
    # Libera el recurso del video.

    cv2.destroyAllWindows()
    # Cierra ventanas OpenCV.


if __name__ == "__main__":
    main()
    # Punto de entrada del script.
