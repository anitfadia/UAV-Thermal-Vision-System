
import cv2
import numpy as np
import os
import glob
import time

# Dataset paths
BASE = os.path.join(
    "dataset", "images", "thermal dataset", "test"
)
IMAGE_DIR = os.path.join(BASE, "images")
LABEL_DIR = os.path.join(BASE, "labels")

# Search ranges: adjust after the first run if needed
CLIP_LIMITS = [2.0]
TILE_SIZES = [8]
THRESHOLDS = [120, 140, 160]
MIN_AREAS = [600, 800]

IOU_THRESHOLD = 0.5


def load_labels(path, width, height):
    boxes = []

    if not os.path.exists(path):
        return boxes

    with open(path, "r") as f:
        for line in f:
            values = line.strip().split()
            if len(values) != 5:
                continue

            cls, xc, yc, bw, bh = map(float, values)

            # Dataset class 0 represents the human class
            if int(cls) != 0:
                continue

            x1 = int((xc - bw / 2) * width)
            y1 = int((yc - bh / 2) * height)
            x2 = int((xc + bw / 2) * width)
            y2 = int((yc + bh / 2) * height)

            boxes.append([
                max(0, x1), max(0, y1),
                min(width - 1, x2), min(height - 1, y2)
            ])

    return boxes


def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = max(0, box1[2] - box1[0]) * max(
        0, box1[3] - box1[1]
    )
    area2 = max(0, box2[2] - box2[0]) * max(
        0, box2[3] - box2[1]
    )

    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0


def detect(image, clahe, threshold, min_area):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    enhanced = clahe.apply(gray)

    # Reduce small noise while preserving edges
    filtered = cv2.GaussianBlur(enhanced, (3, 3), 0)

    _, mask = cv2.threshold(
        filtered, threshold, 255, cv2.THRESH_BINARY
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (3, 3)
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, kernel, iterations=1
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, kernel, iterations=2
    )

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    predictions = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        if h < 40 or w < 15:
            continue

        aspect_ratio = h / float(w)
        if not 1.0 <= aspect_ratio <= 4.5:
            continue

        predictions.append([x, y, x + w, y + h])

    return predictions


def match_boxes(predictions, ground_truth):
    # Greedy one-to-one matching by highest IoU
    candidates = []

    for pi, pred in enumerate(predictions):
        for gi, gt in enumerate(ground_truth):
            score = iou(pred, gt)
            if score >= IOU_THRESHOLD:
                candidates.append((score, pi, gi))

    candidates.sort(reverse=True)

    used_pred = set()
    used_gt = set()
    tp = 0

    for score, pi, gi in candidates:
        if pi not in used_pred and gi not in used_gt:
            tp += 1
            used_pred.add(pi)
            used_gt.add(gi)

    fp = len(predictions) - tp
    fn = len(ground_truth) - tp
    return tp, fp, fn


def main():
    image_paths = sorted(
        glob.glob(os.path.join(IMAGE_DIR, "*"))
    )
    image_paths = [
        p for p in image_paths
        if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]

    if not image_paths:
        print("No test images found. Check IMAGE_DIR.")
        return

    print("Test images:", len(image_paths))
    results = []

    for clip in CLIP_LIMITS:
        for tile in TILE_SIZES:
            clahe = cv2.createCLAHE(
                clipLimit=clip,
                tileGridSize=(tile, tile)
            )

            # Cache enhanced images for this CLAHE setting
            cache = []
            total_gt = 0

            for image_path in image_paths:
                image = cv2.imread(image_path)
                if image is None:
                    continue

                h, w = image.shape[:2]
                label_path = os.path.join(
                    LABEL_DIR,
                    os.path.splitext(os.path.basename(image_path))[0]
                    + ".txt"
                )

                gt = load_labels(label_path, w, h)
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                enhanced = clahe.apply(gray)
                filtered = cv2.GaussianBlur(
                    enhanced, (3, 3), 0
                )

                cache.append((image_path, filtered, gt))
                total_gt += len(gt)

            for threshold in THRESHOLDS:
                for min_area in MIN_AREAS:
                    tp = fp = fn = 0
                    start = time.time()

                    for image_path, filtered, gt in cache:
                        _, mask = cv2.threshold(
                            filtered, threshold, 255,
                            cv2.THRESH_BINARY
                        )

                        kernel = cv2.getStructuringElement(
                            cv2.MORPH_ELLIPSE, (3, 3)
                        )
                        mask = cv2.morphologyEx(
                            mask, cv2.MORPH_OPEN, kernel
                        )
                        mask = cv2.morphologyEx(
                            mask, cv2.MORPH_CLOSE,
                            kernel, iterations=2
                        )

                        contours, _ = cv2.findContours(
                            mask, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE
                        )

                        preds = []
                        for c in contours:
                            if cv2.contourArea(c) < min_area:
                                continue

                            x, y, w, h = cv2.boundingRect(c)
                            if h < 40 or w < 15:
                                continue

                            ratio = h / float(w)
                            if 1.0 <= ratio <= 4.5:
                                preds.append(
                                    [x, y, x + w, y + h]
                                )

                        a, b, c = match_boxes(preds, gt)
                        tp += a
                        fp += b
                        fn += c

                    precision = tp / (tp + fp) if tp + fp else 0
                    recall = tp / (tp + fn) if tp + fn else 0
                    f1 = (
                        2 * precision * recall / (precision + recall)
                        if precision + recall else 0
                    )

                    results.append({
                        "f1": f1,
                        "precision": precision,
                        "recall": recall,
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                        "clip": clip,
                        "tile": tile,
                        "threshold": threshold,
                        "min_area": min_area,
                        "seconds": time.time() - start
                    })

            print(f"Finished CLAHE clip={clip}, tile={tile}")

    results.sort(key=lambda r: r["f1"], reverse=True)

    print("\n========== TOP 10 CLAHE SETTINGS ==========")
    for r in results[:10]:
        print(
            f"clip={r['clip']}, tile={r['tile']}, "
            f"threshold={r['threshold']}, area={r['min_area']} | "
            f"TP={r['tp']} FP={r['fp']} FN={r['fn']} | "
            f"P={r['precision']:.4f} R={r['recall']:.4f} "
            f"F1={r['f1']:.4f} | "
            f"time={r['seconds']:.1f}s"
        )

    with open("clahe_tuning_results.csv", "w") as f:
        f.write(
            "clip,tile,threshold,min_area,tp,fp,fn,"
            "precision,recall,f1,seconds\n"
        )
        for r in results:
            f.write(
                f"{r['clip']},{r['tile']},{r['threshold']},"
                f"{r['min_area']},{r['tp']},{r['fp']},{r['fn']},"
                f"{r['precision']:.6f},{r['recall']:.6f},"
                f"{r['f1']:.6f},{r['seconds']:.2f}\n"
            )

    print("\nSaved all results to clahe_tuning_results.csv")


if __name__ == "__main__":
    main()