import cv2
import json
import numpy as np
from pathlib import Path

# Function to crop tightly around ink given a normalized bbox [ymin, xmin, ymax, xmax] (0-1000)
def crop_from_bbox(img_path, bbox_1000, save_path, padding=10):
    img = cv2.imread(img_path)
    if img is None: return False
    h, w = img.shape[:2]
    
    ymin, xmin, ymax, xmax = bbox_1000
    y1 = max(0, int(ymin * h / 1000) - padding)
    y2 = min(h, int(ymax * h / 1000) + padding)
    x1 = max(0, int(xmin * w / 1000) - padding)
    x2 = min(w, int(xmax * w / 1000) + padding)
    
    roi = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)
    pts = np.argwhere(thresh > 0)
    if len(pts) < 30:
        return False
    
    ry1, rx1 = pts.min(axis=0)
    ry2, rx2 = pts.max(axis=0)
    
    fy1 = max(0, y1 + ry1 - 4)
    fy2 = min(h, y1 + ry2 + 5)
    fx1 = max(0, x1 + rx1 - 4)
    fx2 = min(w, x1 + rx2 + 5)
    
    crop = img[fy1:fy2, fx1:fx2]
    crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    ch, cw = crop.shape[:2]
    
    # Transparent white stroke transform (cut-to-cut super-resolution style)
    ink_mask = np.clip((245.0 - crop_gray.astype(float)) / 160.0, 0.0, 1.0)
    scale = 2.0
    up_w = int(cw * scale)
    up_h = int(ch * scale)
    up_ink = cv2.resize(ink_mask, (up_w, up_h), interpolation=cv2.INTER_LANCZOS4)
    smooth_ink = cv2.GaussianBlur(up_ink, (3, 3), 0.5)
    ink_mask = np.clip((smooth_ink - 0.06) / 0.88, 0.0, 1.0)
    
    out = np.zeros((up_h, up_w, 4), dtype=np.uint8)
    out[:, :, 0] = 255
    out[:, :, 1] = 255
    out[:, :, 2] = 255
    out[:, :, 3] = (ink_mask * 255).astype(np.uint8)
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(save_path, out)
    return True, (fx1, fy1, fx2, fy2)

print("crop_from_bbox defined successfully.")
