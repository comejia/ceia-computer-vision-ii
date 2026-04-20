# -*- coding: utf-8 -*-

#%%

import sys
import time
import platform
import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image

MODEL_ID = "apple/DepthPro-hf"
SOURCE = 0
WIDTH, HEIGHT = 640, 480
USE_HALF = True
MAX_FPS = 0.0
CMAP = "turbo"
HFLIP = False

NORMALIZE_MODE = "absolute"
ABS_MIN_M = 0.2
ABS_MAX_M = 10.0

CMAPS = {
    "turbo": cv2.COLORMAP_TURBO,
    "jet": cv2.COLORMAP_JET,
    "inferno": cv2.COLORMAP_INFERNO,
    "plasma": cv2.COLORMAP_PLASMA,
    "magma": cv2.COLORMAP_MAGMA,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "hot": cv2.COLORMAP_HOT,
    "cool": cv2.COLORMAP_COOL,
    "ocean": cv2.COLORMAP_OCEAN,
}

def build_model(model_id, device, half):
    image_processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForDepthEstimation.from_pretrained(model_id)
    model.to(device)
    if half:
        model.half()
    model.eval()
    return image_processor, model

def infer_depth_meter(image_processor, model, frame_bgr, device):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(frame_rgb)
    inputs = image_processor(images=pil, return_tensors="pt").to(device)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device == "cuda")):
        outputs = model(**inputs)
    post = image_processor.post_process_depth_estimation(
        outputs, target_sizes=[(pil.height, pil.width)]
    )[0]["predicted_depth"]
    return post.float()

def normalize_for_viz(depth_m, mode="relative", vmin=0.2, vmax=10.0):
    if mode == "absolute":
        d = depth_m.clamp(vmin, vmax)
        d = (d - vmin) / max(vmax - vmin, 1e-8)
    else:
        dmin = depth_m.min()
        dmax = depth_m.max()
        d = (depth_m - dmin) / (dmax - dmin + 1e-8)
    gray = (d.detach().cpu().numpy() * 255.0).astype(np.uint8)
    return gray

def colorize(gray, cmap_name):
    cmap = CMAPS.get(cmap_name, cv2.COLORMAP_TURBO)
    return cv2.applyColorMap(gray, cmap)

def compose_side_by_side(rgb, depth_color):
    h, w = rgb.shape[:2]
    depth_color = cv2.resize(depth_color, (w, h), interpolation=cv2.INTER_NEAREST)
    return np.hstack([rgb, depth_color])

def put_text(img, text, pos, scale=0.7, color=(255, 255, 255)):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)

def open_camera(source, width, height):
    sysname = platform.system().lower()
    backends = []
    if "windows" in sysname:
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
    elif "linux" in sysname:
        backends = [cv2.CAP_V4L2, cv2.CAP_ANY]
    else:
        backends = [cv2.CAP_ANY]
    cap = None
    for be in backends:
        cap = cv2.VideoCapture(source, be)
        if cap.isOpened():
            break
        if cap is not None:
            cap.release()
            cap = None
    if cap is None or not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap

def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    half = USE_HALF and (device == "cuda")
    torch.backends.cudnn.benchmark = True

    cv2.namedWindow("Depth Pro | Izquierda: RGB  |  Derecha: Profundidad", cv2.WINDOW_NORMAL)
    placeholder = np.zeros((HEIGHT, 2*WIDTH, 3), dtype=np.uint8)
    put_text(placeholder, "Inicializando...", (20, 40))
    cv2.imshow("Depth Pro | Izquierda: RGB  |  Derecha: Profundidad", placeholder)
    cv2.waitKey(1)

    cap = open_camera(SOURCE, WIDTH, HEIGHT)
    if cap is None:
        print("No se pudo abrir la cámara. Probá con SOURCE=1 o verificá permisos/backend de OpenCV.")
        return

    proc, model = build_model(MODEL_ID, device, half)

    ema_fps, t_prev = None, time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("No se pudo leer frame de la cámara.")
            break

        if HFLIP:
            frame = cv2.flip(frame, 1)

        if MAX_FPS > 0:
            t_now = time.time()
            dt = t_now - t_prev
            if dt < 1.0 / MAX_FPS:
                time.sleep((1.0 / MAX_FPS) - dt)
            t_prev = time.time()

        frame = cv2.resize(frame, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)

        t0 = time.time()
        depth_m = infer_depth_meter(proc, model, frame, device)
        depth_gray = normalize_for_viz(depth_m, NORMALIZE_MODE, ABS_MIN_M, ABS_MAX_M)
        depth_color = colorize(depth_gray, CMAP)
        fused = compose_side_by_side(frame, depth_color)
        t1 = time.time()

        fps = 1.0 / max(t1 - t0, 1e-6)
        ema_fps = fps if ema_fps is None else 0.9 * ema_fps + 0.1 * fps

        midx = fused.shape[1] // 2
        put_text(fused, "RGB", (15, 30))
        put_text(fused, f"{MODEL_ID} | {NORMALIZE_MODE}", (midx + 15, 30))
        put_text(fused, f"FPS: {ema_fps:.1f}", (15, fused.shape[0] - 20), 0.8, (20, 220, 20))

        cv2.imshow("Depth Pro | Izquierda: RGB  |  Derecha: Profundidad", fused)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()
