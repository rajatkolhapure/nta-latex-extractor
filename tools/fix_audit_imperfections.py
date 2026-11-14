import cv2
import json
import numpy as np

# 1. Fix 140_17
# Question text: rows 0..65
# Circuit diagram: rows 70..245
# Options: rows 250..350
img_140_17 = cv2.imread("downloads/question_images/17_20190919115224.PNG")
if img_140_17 is not None:
    out_path = "downloads/diagrams/Physics/diagram_17_20190919115224.png"
    gray = cv2.cvtColor(img_140_17, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
    th[:68, :] = 0   # remove question text
    th[246:, :] = 0  # remove options!
    pts = np.argwhere(th > 0)
    if len(pts):
        y1, x1 = pts.min(axis=0)
        y2, x2 = pts.max(axis=0)
        crop = img_140_17[y1:y2+1, x1:x2+1]
        cg = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        ink_mask = np.clip((245.0 - cg.astype(float)) / 160.0, 0.0, 1.0)
        up_w = int(crop.shape[1] * 2.0)
        up_h = int(crop.shape[0] * 2.0)
        up_ink = cv2.resize(ink_mask, (up_w, up_h), interpolation=cv2.INTER_LANCZOS4)
        out = np.zeros((up_h, up_w, 4), dtype=np.uint8)
        out[:, :, :3] = 255
        out[:, :, 3] = (up_ink * 255).astype(np.uint8)
        cv2.imwrite(out_path, out)
        print(f"[+] Fixed 140_17: cropped strictly circuit (y={y1}..{y2}, x={x1}..{x2}), options excluded!")

# 2. Fix 148_14
# Capacitor with 4 dielectrics
img_148_14 = cv2.imread("downloads/question_images/14_20191003150620.JPG")
if img_148_14 is not None:
    h, w = img_148_14.shape[:2]
    out_path = "downloads/diagrams/Physics/diagram_14_20191003150620.png"
    gray = cv2.cvtColor(img_148_14, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY_INV)
    # Exclude question text at top and options at bottom
    # Text is top ~130px, options are bottom ~250px
    th[:130, :] = 0
    th[int(h*0.65):, :] = 0
    pts = np.argwhere(th > 0)
    if len(pts):
        y1, x1 = pts.min(axis=0)
        y2, x2 = pts.max(axis=0)
        crop = img_148_14[y1:y2+1, x1:x2+1]
        cg = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        ink_mask = np.clip((245.0 - cg.astype(float)) / 160.0, 0.0, 1.0)
        up_w = int(crop.shape[1] * 2.0)
        up_h = int(crop.shape[0] * 2.0)
        up_ink = cv2.resize(ink_mask, (up_w, up_h), interpolation=cv2.INTER_LANCZOS4)
        out = np.zeros((up_h, up_w, 4), dtype=np.uint8)
        out[:, :, :3] = 255
        out[:, :, 3] = (up_ink * 255).astype(np.uint8)
        cv2.imwrite(out_path, out)
        print(f"[+] Fixed 148_14: cropped strictly capacitor diagram (y={y1}..{y2}, x={x1}..{x2})")

# 3. Fix 155_16 & 161_9 (False body diagrams - they are option diagrams!)
# In 155_16 and 161_9, the question has no body figure, only option diagrams!
d = json.load(open('data/re_extracted_flagged.json', encoding='utf-8'))
for q in d:
    k = f"{q['paperId']}_{q['questionNumber']}"
    if k in ['155_16', '161_9']:
        q['hasDiagram'] = False
        q['diagramPath'] = None
        q['optionsHaveDiagrams'] = True
        print(f"[+] Fixed {k}: removed false body diagram, kept option diagrams")
    if k == '133_25':
        # Add $ delimiters to options
        for opt in q.get('latexOptions', []):
            txt = opt.get('latex', '')
            if not txt.startswith('$'):
                opt['latex'] = f"${txt}$"
        print("[+] Fixed 133_25: added $ delimiters to all options")

with open('data/re_extracted_flagged.json', 'w', encoding='utf-8') as f:
    json.dump(d, f, indent=2, ensure_ascii=False)
