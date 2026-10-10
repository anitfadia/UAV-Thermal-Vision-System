import cv2
import matplotlib.pyplot as plt
import time
from detection import detect_candidates


# -----------------------------
# Load thermal image
# -----------------------------

image = cv2.imread(
    "dataset/images/test.jpg",
    cv2.IMREAD_GRAYSCALE
)

if image is None:
    print("ERROR: Could not load image")
    exit()

print("Image loaded successfully!")


# -----------------------------
# 1. Normalization
# -----------------------------

normalized = cv2.normalize(
    image,
    None,
    0,
    255,
    cv2.NORM_MINMAX
)


# -----------------------------
# 2. Global Histogram Equalization
# -----------------------------

global_equalized = cv2.equalizeHist(normalized)


# -----------------------------
# 3. CLAHE
# -----------------------------

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

clahe_image = clahe.apply(normalized)


# -----------------------------
# 4. CLAHE + Bilateral Filter
# -----------------------------

filtered = cv2.bilateralFilter(
    clahe_image,
    9,
    75,
    75
)


# -----------------------------
# Function to draw detections
# -----------------------------

def draw_detections(image, detections, title):

    result = cv2.cvtColor(
        image,
        cv2.COLOR_GRAY2RGB
    )

    for i, (x, y, w, h) in enumerate(detections):

        # Bounding box
        cv2.rectangle(
            result,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        # Centroid
        cx = x + w // 2
        cy = y + h // 2

        cv2.circle(
            result,
            (cx, cy),
            5,
            (255, 0, 0),
            -1
        )

        # Label
        cv2.putText(
            result,
            "Candidate " + str(i + 1),
            (x, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

    return result


# -----------------------------
# Run detection on each method
# -----------------------------

print("\n--- Normalized Image ---")
start = time.time()
detections_normalized = detect_candidates(normalized)
time_normalized = time.time() - start


print("\n--- Global Histogram Equalization ---")
start = time.time()
detections_global = detect_candidates(global_equalized)
time_global = time.time() - start


print("\n--- CLAHE ---")
start = time.time()
detections_clahe = detect_candidates(clahe_image)
time_clahe = time.time() - start


print("\n--- CLAHE + Bilateral Filter ---")
start = time.time()
detections_filtered = detect_candidates(filtered)
time_filtered = time.time() - start


# -----------------------------
# Draw results
# -----------------------------

result1 = draw_detections(
    normalized,
    detections_normalized,
    "Normalized"
)

result2 = draw_detections(
    global_equalized,
    detections_global,
    "Global Equalization"
)

result3 = draw_detections(
    clahe_image,
    detections_clahe,
    "CLAHE"
)

result4 = draw_detections(
    filtered,
    detections_filtered,
    "CLAHE + Bilateral"
)


# -----------------------------
# Display comparison
# -----------------------------

plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.imshow(result1)
plt.title(
    "Normalized\nCandidates: "
    + str(len(detections_normalized))
)
plt.axis("off")


plt.subplot(2, 2, 2)
plt.imshow(result2)
plt.title(
    "Global Equalization\nCandidates: "
    + str(len(detections_global))
)
plt.axis("off")


plt.subplot(2, 2, 3)
plt.imshow(result3)
plt.title(
    "CLAHE\nCandidates: "
    + str(len(detections_clahe))
)
plt.axis("off")


plt.subplot(2, 2, 4)
plt.imshow(result4)
plt.title(
    "CLAHE + Bilateral\nCandidates: "
    + str(len(detections_filtered))
)
plt.axis("off")


plt.tight_layout()

# Save comparison
plt.savefig(
    "output/clahe_comparison.jpg",
    dpi=300
)

print("\nComparison saved to:")
print("output/clahe_comparison.jpg")

plt.show()


# -----------------------------
# Print processing times
# -----------------------------

print("\n===== PROCESSING TIMES =====")

print(
    "Normalized:",
    round(time_normalized, 4),
    "seconds"
)

print(
    "Global Equalization:",
    round(time_global, 4),
    "seconds"
)

print(
    "CLAHE:",
    round(time_clahe, 4),
    "seconds"
)

print(
    "CLAHE + Bilateral:",
    round(time_filtered, 4),
    "seconds"
)

print("\nSTAGE 4 FINISHED")