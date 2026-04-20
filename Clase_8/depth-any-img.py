# -*- coding: utf-8 -*-

#%%

import torch
import cv2
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

MODEL_ID = "depth-anything/Depth-Anything-V2-Large-hf"

IMAGE_PATH = "test5.jpg"

OUT_GRAY = "depth_gray_3.png"
OUT_COLOR = "depth_color_3.png"

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoImageProcessor.from_pretrained(MODEL_ID)
model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID).to(device)
model.eval()

image = Image.open(IMAGE_PATH).convert("RGB")

inputs = processor(images=image, return_tensors="pt").to(device)
with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device == "cuda")):
    outputs = model(**inputs)

post = processor.post_process_depth_estimation(
    outputs, target_sizes=[(image.height, image.width)]
)[0]["predicted_depth"]

depth = post.float()
depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
depth_gray = (depth.cpu().numpy() * 255.0).astype(np.uint8)

depth_color = cv2.applyColorMap(depth_gray, cv2.COLORMAP_TURBO)

cv2.imwrite(OUT_GRAY, depth_gray)
cv2.imwrite(OUT_COLOR, depth_color)

cv2.imshow("Imagen original", cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
cv2.imshow("Profundidad (color)", depth_color)
cv2.waitKey(0)
cv2.destroyAllWindows()


