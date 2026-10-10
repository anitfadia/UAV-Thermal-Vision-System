
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

IMAGE_PATH = (
    ROOT / "dataset" / "images" / "thermal dataset"
    / "test" / "images" / "190001.jpg"
)
LABEL_PATH = (
    ROOT / "dataset" / "images" / "thermal dataset"
    / "test" / "labels" / "190001.txt"
)

image = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")

height, width = image.shape

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

# Load ground-truth boxes
ground_truth = []

for line in LABEL_PATH.read_text().splitlines():
    parts = line.split()
    if len(parts) < 5 or int(parts[0]) != 0:
        continue

    _, xc, yc, bw, bh = map(float, parts[:5])

    x1 = int((xc - bw / 2) * width)
    y1 = int((yc - bh / 2) * height)
    x2 = int((xc + bw / 2) * width)
    y2 = int((yc + bh / 2) * height)

    ground_truth.append((x1, y1, x2, y2))


def update(_=None):
    threshold = cv2.getTrackbarPos("Threshold", "Threshold Tuner")
    use_clahe = cv2.getTrackbarPos("CLAHE: 0=Off 1=On", "Threshold Tuner")

    processed = clahe.apply(image) if use_clahe else image

    _, mask = cv2.threshold(
        processed, threshold, 255, cv2.THRESH_BINARY
    )

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    display = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    candidates = 0

    # Red = detector candidates
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        ratio = h / max(w, 1)

        if (
            area >= 300
            and h > 40
            and w > 15
            and 1.0 <= ratio <= 4.5
        ):
            cv2.rectangle(
                display, (x, y), (x + w, y + h), (0, 0, 255), 2
            )
            candidates += 1

    # Green = ground-truth human boxes
    for x1, y1, x2, y2 in ground_truth:
        cv2.rectangle(
            display, (x1, y1), (x2, y2), (0, 255, 0), 2
        )

    method = "CLAHE ON" if use_clahe else "CLAHE OFF"
    cv2.putText(
        display,
        f"{method} | Threshold: {threshold} | Candidates: {candidates}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.imshow("Threshold Tuner", display)
    cv2.imshow("Threshold Mask", mask)


cv2.namedWindow("Threshold Tuner")
cv2.createTrackbar("Threshold", "Threshold Tuner", 180, 255, update)
cv2.createTrackbar("CLAHE: 0=Off 1=On", "Threshold Tuner", 0, 1, update)

update()

print("Adjust the threshold slider and toggle CLAHE.")
print("Close the image windows to exit.")

cv2.waitKey(0)
cv2.destroyAllWindows()