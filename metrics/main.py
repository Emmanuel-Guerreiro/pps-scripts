import os
import cv2
import json
import numpy as np
from glob import glob

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
# Order must match YOLO class indices
CLASSES = ['no-eyewear', 'no-gloves', 'no-mask', 'person']

GT_DIR = "/home/emma/facultad/pps/parte2/to-test-metrics/ppe-detect.v5i.yolov11/test/labels"
PRED_DIR = "/home/emma/facultad/pps/parte2/to-test-metrics/validacion-ppe-detect.v5i.yolov11/predictions"
IMG_DIR = "/home/emma/facultad/pps/parte2/to-test-metrics/ppe-detect.v5i.yolov11/test/images"

IOU_THRESHOLD = 0.5



# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------
def yolo_to_xyxy(yolo_box, img_w, img_h):
    cls, cx, cy, w, h = yolo_box
    cx *= img_w
    cy *= img_h
    w  *= img_w
    h  *= img_h
    x1 = int(cx - w/2)
    y1 = int(cy - h/2)
    x2 = int(cx + w/2)
    y2 = int(cy + h/2)
    return int(cls), x1, y1, x2, y2


def iou(boxA, boxB):
    x1 = max(boxA[0], boxB[0])
    y1 = max(boxA[1], boxB[1])
    x2 = min(boxA[2], boxB[2])
    y2 = min(boxA[3], boxB[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0

    areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
    areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0.0


def load_ground_truth(gt_path, img_w, img_h):
    boxes = []
    with open(gt_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            parts = list(map(float, line.strip().split()))
            cls, cx, cy, w, h = parts[:5]
            cls, x1, y1, x2, y2 = yolo_to_xyxy((cls, cx, cy, w, h), img_w, img_h)
            if 0 <= cls < len(CLASSES):
                boxes.append((cls, x1, y1, x2, y2))
    return boxes


def load_predictions(pred_path, img_w, img_h):
    preds = []

    # ----------- JSON (Roboflow) -----------
    if pred_path.endswith(".json"):
        data = json.load(open(pred_path))
        for obj in data.get("predictions", []):
            if "class_id" not in obj:
                continue

            cls_id = int(obj["class_id"])
            if cls_id < 0 or cls_id >= len(CLASSES):
                continue

            x = obj["x"]
            y = obj["y"]
            w = obj["width"]
            h = obj["height"]

            x1 = int(x - w/2)
            y1 = int(y - h/2)
            x2 = int(x + w/2)
            y2 = int(y + h/2)
            conf = float(obj.get("confidence", 1.0))

            preds.append((cls_id, x1, y1, x2, y2, conf))

        return preds

    # ----------- YOLO TXT predictions ----------
    if pred_path.endswith(".txt"):
        with open(pred_path) as f:
            for line in f:
                if not line.strip():
                    continue
                cls, cx, cy, w, h, conf = map(float, line.split())
                cls, x1, y1, x2, y2 = yolo_to_xyxy((cls, cx, cy, w, h), img_w, img_h)
                if 0 <= cls < len(CLASSES):
                    preds.append((cls, x1, y1, x2, y2, conf))
        return preds

    return preds


def compute_metrics(TP, FP, FN, ious):
    precision = TP / (TP + FP + 1e-6)
    recall    = TP / (TP + FN + 1e-6)
    miou      = np.mean(ious) if len(ious) else 0.0
    map50     = precision
    return miou, precision, recall, map50


# ------------------------------------------------------------
# METRICS STORAGE
# ------------------------------------------------------------
class_metrics = {
    cls_id: {"TP":0, "FP":0, "FN":0, "ious": []}
    for cls_id in range(len(CLASSES))
}


# ------------------------------------------------------------
# EVALUATION
# ------------------------------------------------------------
for gt_file in sorted(glob(os.path.join(GT_DIR, "*.txt"))):
    fname = os.path.basename(gt_file).replace(".txt", "")

    # pick JPG or PNG
    img_path = None
    for ext in [".jpg", ".png", ".jpeg"]:
        p = os.path.join(IMG_DIR, fname + ext)
        if os.path.exists(p):
            img_path = p
            break
    if img_path is None:
        print(f"[WARN] Missing image: {fname}")
        continue

    # predictions
    pred_path = None
    if os.path.exists(os.path.join(PRED_DIR, fname + ".json")):
        pred_path = os.path.join(PRED_DIR, fname + ".json")
    elif os.path.exists(os.path.join(PRED_DIR, fname + ".txt")):
        pred_path = os.path.join(PRED_DIR, fname + ".txt")
    else:
        print(f"[WARN] Missing prediction for: {fname}")
        continue

    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    gt_boxes   = load_ground_truth(gt_file, w, h)
    pred_boxes = load_predictions(pred_path, w, h)

    # Group GT indices per class
    gt_by_class = {c: set() for c in range(len(CLASSES))}
    for idx, (cls, *_coords) in enumerate(gt_boxes):
        gt_by_class[int(cls)].add(idx)

    # ---- CLASS-SPECIFIC MATCHING ----
    for cls_id in range(len(CLASSES)):
        preds = [p for p in pred_boxes if p[0] == cls_id]
        gt_ids = gt_by_class[cls_id]

        preds = sorted(preds, key=lambda x: x[5], reverse=True)

        matched_gt = set()

        for (_, px1, py1, px2, py2, conf) in preds:
            best_i = 0
            best_gt = -1

            for g in gt_ids:
                if g in matched_gt:
                    continue
                _, gx1, gy1, gx2, gy2 = gt_boxes[g]
                iov = iou((px1, py1, px2, py2), (gx1, gy1, gx2, gy2))
                if iov > best_i:
                    best_i = iov
                    best_gt = g

            if best_gt != -1 and best_i >= IOU_THRESHOLD:
                matched_gt.add(best_gt)
                class_metrics[cls_id]["TP"] += 1
                class_metrics[cls_id]["ious"].append(best_i)
            else:
                class_metrics[cls_id]["FP"] += 1

        # FN = GTs not matched
        class_metrics[cls_id]["FN"] += len(gt_ids) - len(matched_gt)


# ------------------------------------------------------------
# GLOBAL METRICS (SUM OF ALL CLASSES)
# ------------------------------------------------------------
global_TP = sum(m["TP"] for m in class_metrics.values())
global_FP = sum(m["FP"] for m in class_metrics.values())
global_FN = sum(m["FN"] for m in class_metrics.values())
global_ious = [iou for m in class_metrics.values() for iou in m["ious"]]

gm_iou, gm_prec, gm_rec, gm_map = compute_metrics(global_TP, global_FP, global_FN, global_ious)


# ------------------------------------------------------------
# PRINT RESULTS
# ------------------------------------------------------------
print("\n====================== GLOBAL METRICS ======================")
print(f"mIoU:      {gm_iou:.4f}")
print(f"Precision: {gm_prec:.4f}")
print(f"Recall:    {gm_rec:.4f}")
print(f"mAP@50:    {gm_map:.4f}")

print("\n====================== PER-CLASS METRICS ======================")
for cls_id, cls_name in enumerate(CLASSES):
    m = class_metrics[cls_id]
    miou, prec, rec, map50 = compute_metrics(m["TP"], m["FP"], m["FN"], m["ious"])
    print(f"\nClass: {cls_name}")
    print(f"  TP:        {m['TP']}")
    print(f"  FP:        {m['FP']}")
    print(f"  FN:        {m['FN']}")
    print(f"  mIoU:      {miou:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  mAP@50:    {map50:.4f}")
