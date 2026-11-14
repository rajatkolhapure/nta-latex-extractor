import cv2
import json
import numpy as np

d = json.load(open('data/re_extracted_flagged.json', encoding='utf-8'))
q_map = {f"{q['paperId']}_{q['questionNumber']}": q for q in d}

# 150_15 (dipole diagram)
# Image 610x415. Question text rows 0..90. Diagram rows 95..220. Options row 235+.
img_150_15 = cv2.imread("downloads/question_images/15_20190930104507.JPG")
if img_150_15 is not None:
    out_path = "downloads/diagrams/Physics/diagram_15_20190930104507.png"
    gray = cv2.cvtColor(img_150_15, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
    th[:90, :] = 0
    th[225:, :] = 0
    pts = np.argwhere(th > 0)
    if len(pts):
        y1, x1 = pts.min(axis=0)
        y2, x2 = pts.max(axis=0)
        crop = img_150_15[y1:y2+1, x1:x2+1]
        cg = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        ink_mask = np.clip((245.0 - cg.astype(float)) / 160.0, 0.0, 1.0)
        up_w = int(crop.shape[1] * 2.0)
        up_h = int(crop.shape[0] * 2.0)
        up_ink = cv2.resize(ink_mask, (up_w, up_h), interpolation=cv2.INTER_LANCZOS4)
        out = np.zeros((up_h, up_w, 4), dtype=np.uint8)
        out[:, :, :3] = 255
        out[:, :, 3] = (up_ink * 255).astype(np.uint8)
        cv2.imwrite(out_path, out)
        q_map["150_15"]["hasDiagram"] = True
        q_map["150_15"]["diagramPath"] = out_path
        q_map["150_15"]["diagramBBox"] = [int(x1), int(y1), int(x2), int(y2)]
        print(f"[+] Cleaned 150_15 dipole diagram crop: {x1},{y1} to {x2},{y2}")

# 153_28 (LC / AC circuit or resonance diagram)
# Image 491x346.
img_153_28 = cv2.imread("downloads/question_images/28_20191003112236.JPG")
if img_153_28 is not None:
    out_path = "downloads/diagrams/Physics/diagram_28_20191003112236.png"
    gray = cv2.cvtColor(img_153_28, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
    # Check where diagram sits (exclude top text rows 0..70, bottom options 220+)
    th[:70, :] = 0
    th[240:, :] = 0
    pts = np.argwhere(th > 0)
    if len(pts) > 50:
        y1, x1 = pts.min(axis=0)
        y2, x2 = pts.max(axis=0)
        crop = img_153_28[y1:y2+1, x1:x2+1]
        cg = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        ink_mask = np.clip((245.0 - cg.astype(float)) / 160.0, 0.0, 1.0)
        up_w = int(crop.shape[1] * 2.0)
        up_h = int(crop.shape[0] * 2.0)
        up_ink = cv2.resize(ink_mask, (up_w, up_h), interpolation=cv2.INTER_LANCZOS4)
        out = np.zeros((up_h, up_w, 4), dtype=np.uint8)
        out[:, :, :3] = 255
        out[:, :, 3] = (up_ink * 255).astype(np.uint8)
        cv2.imwrite(out_path, out)
        q_map["153_28"]["hasDiagram"] = True
        q_map["153_28"]["diagramPath"] = out_path
        q_map["153_28"]["diagramBBox"] = [int(x1), int(y1), int(x2), int(y2)]
        print(f"[+] Cropped 153_28 circuit diagram: {x1},{y1} to {x2},{y2}")

with open("data/re_extracted_flagged.json", "w", encoding="utf-8") as f:
    json.dump(d, f, indent=2, ensure_ascii=False)
