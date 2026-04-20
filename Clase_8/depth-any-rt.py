import time
import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
SOURCE = 0
WIDTH, HEIGHT = 640, 480
USE_HALF = True
MAX_FPS = 0.0
CMAP = "ocean"
HFLIP = False

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

def to_depth(image_processor, model, frame_bgr, device):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(frame_rgb)
    inputs = image_processor(images=pil, return_tensors="pt").to(device)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device == "cuda")):
        outputs = model(**inputs)
    post = image_processor.post_process_depth_estimation(
        outputs, target_sizes=[(pil.height, pil.width)]
    )[0]["predicted_depth"]
    depth = (post - post.min()) / (post.max() - post.min() + 1e-8)
    depth_np = (depth.detach().cpu().numpy() * 255.0).astype(np.uint8)
    return depth_np

def colorize(depth_gray, cmap_name):
    cmap = CMAPS.get(cmap_name, 'ocean')
    return cv2.applyColorMap(depth_gray, cmap)

def compose_side_by_side(rgb, depth_color):
    h, w = rgb.shape[:2]
    depth_color = cv2.resize(depth_color, (w, h), interpolation=cv2.INTER_NEAREST)
    return np.hstack([rgb, depth_color])

def overlay_text(img, text, pos, scale=0.7, color=(255, 255, 255)):
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)

def run():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    half = USE_HALF and (device == "cuda")
    torch.backends.cudnn.benchmark = True

    proc, model = build_model(MODEL_ID, device, half)

    cap = cv2.VideoCapture(SOURCE)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    ema_fps, t_prev = None, time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
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
        dmap = to_depth(proc, model, frame, device)
        dcol = colorize(dmap, CMAP)
        fused = compose_side_by_side(frame, dcol)
        t1 = time.time()

        fps = 1.0 / max(t1 - t0, 1e-6)
        ema_fps = fps if ema_fps is None else 0.9 * ema_fps + 0.1 * fps
        w = fused.shape[1] // 2
        overlay_text(fused, "RGB", (15, 30))
        overlay_text(fused, MODEL_ID, (w + 15, 30))
        overlay_text(fused, f"FPS: {ema_fps:.1f}", (15, fused.shape[0] - 20), 0.8, (20, 220, 20))

        cv2.imshow("Depth Anything V2 | Izquierda: RGB  |  Derecha: Profundidad", fused)
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run()
