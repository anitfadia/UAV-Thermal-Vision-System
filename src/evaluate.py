
from pathlib import Path
import time
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TEST_IMAGES = ROOT / "dataset" / "images" / "thermal dataset" / "test" / "images"
TEST_LABELS = ROOT / "dataset" / "images" / "thermal dataset" / "test" / "labels"

IOU_THRESHOLD = 0.50


def load_ground_truth(label_path, image_width, image_height):
    boxes = []

    for line in label_path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue

        class_id, xc, yc, bw, bh = map(float, parts[:5])

        # Dataset sample shows class 0; evaluate that class.
        if int(class_id) != 0:
            continue

        x1 = int((xc - bw / 2) * image_width)
        y1 = int((yc - bh / 2) * image_height)
        x2 = int((xc + bw / 2) * image_width)
        y2 = int((yc + bh / 2) * image_height)

        boxes.append([
            max(0, x1), max(0, y1),
            min(image_width, x2), min(image_height, y2)
        ])

    return boxes



def detect_candidates(gray, threshold=180):
    _, mask = cv2.threshold(
        gray, threshold, 255, cv2.THRESH_BINARY
    )

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, kernel
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, kernel
    )

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    predictions = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        ratio = h / max(w, 1)

        if (
            area >= 600
            and h > 40
            and w > 15
            and 1.0 <= ratio <= 4.5
        ):
            predictions.append([x, y, x + w, y + h])

    return predictions

def calculate_iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area_a = max(0, box_a[2] - box_a[0]) * max(
        0, box_a[3] - box_a[1]
    )
    area_b = max(0, box_b[2] - box_b[0]) * max(
        0, box_b[3] - box_b[1]
    )

    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def evaluate_method(use_clahe):
    tp = fp = fn = 0
    total_images = 0
    start = time.perf_counter()

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    for image_path in sorted(TEST_IMAGES.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            continue

        label_path = TEST_LABELS / (image_path.stem + ".txt")
        if not label_path.exists():
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            continue

        height, width = image.shape

        ground_truth = load_ground_truth(
            label_path, width, height
        )

        processed = clahe.apply(image) if use_clahe else image
        threshold = 220 if use_clahe else 180
        predictions = detect_candidates(processed, threshold)

        # Greedy one-to-one matching at IoU >= 0.50.
        matches = []
        for pred_index, pred in enumerate(predictions):
            for gt_index, gt in enumerate(ground_truth):
                iou = calculate_iou(pred, gt)
                if iou >= IOU_THRESHOLD:
                    matches.append((iou, pred_index, gt_index))

        matches.sort(reverse=True)

        used_predictions = set()
        used_ground_truth = set()

        for _, pred_index, gt_index in matches:
            if (
                pred_index not in used_predictions
                and gt_index not in used_ground_truth
            ):
                used_predictions.add(pred_index)
                used_ground_truth.add(gt_index)

        tp += len(used_predictions)
        fp += len(predictions) - len(used_predictions)
        fn += len(ground_truth) - len(used_ground_truth)
        total_images += 1

    elapsed = time.perf_counter() - start

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "method": "CLAHE" if use_clahe else "Original",
        "images": total_images,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "seconds": elapsed,
    }




if __name__ == "__main__":
    if not TEST_IMAGES.exists() or not TEST_LABELS.exists():
        raise FileNotFoundError("Test image or label folder was not found.")

    for threshold in (140, 160, 180, 200):
        print(f"\n========== ORIGINAL: THRESHOLD {threshold} ==========")

        # Temporarily use this threshold for the original images.
        def run_at_threshold(t):
            tp = fp = fn = 0
            start = time.perf_counter()

            for image_path in sorted(TEST_IMAGES.iterdir()):
                if image_path.suffix.lower() not in {
                    ".jpg", ".jpeg", ".png", ".bmp"
                }:
                    continue

                label_path = TEST_LABELS / (image_path.stem + ".txt")
                if not label_path.exists():
                    continue

                image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    continue

                h, w = image.shape
                gt = load_ground_truth(label_path, w, h)
                pred = detect_candidates(image, t)
                
                pairs = []
                for pi, p in enumerate(pred):
                     for gi, g in enumerate(gt):
                          iou = calculate_iou(p, g)
                          if iou >= IOU_THRESHOLD:
                               pairs.append((iou, pi, gi))
                pairs.sort(reverse=True)

                used_p, used_g = set(), set()
                for _, pi, gi in pairs:
                    if pi not in used_p and gi not in used_g:
                        used_p.add(pi)
                        used_g.add(gi)

                tp += len(used_p)
                fp += len(pred) - len(used_p)
                fn += len(gt) - len(used_g)

            precision = tp / (tp + fp) if tp + fp else 0
            recall = tp / (tp + fn) if tp + fn else 0
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall else 0
            )

            print(f"TP={tp}, FP={fp}, FN={fn}")
            print(f"Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
            print(f"Time={time.perf_counter() - start:.2f}s")

        run_at_threshold(threshold)