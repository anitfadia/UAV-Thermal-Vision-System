import cv2

from detection import detect_candidates


# --------------------------------
# LOAD CLAHE IMAGE
# --------------------------------

image = cv2.imread(
    "output/bilateral_filtered.jpg",
    cv2.IMREAD_GRAYSCALE
)

if image is None:
    print("ERROR: Could not load processed image.")
    exit()

print("Processed image loaded successfully!")


# --------------------------------
# HUMAN CANDIDATE DETECTION
# --------------------------------

detections = detect_candidates(image)


# --------------------------------
# CREATE COLOUR IMAGE
# --------------------------------

result = cv2.cvtColor(
    image,
    cv2.COLOR_GRAY2BGR
)


# --------------------------------
# DRAW BOUNDING BOXES
# --------------------------------

for i, (x, y, w, h) in enumerate(detections):

    # Draw bounding box
    cv2.rectangle(
        result,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

    # Calculate centroid
    center_x = x + w // 2
    center_y = y + h // 2

    # Draw centroid
    cv2.circle(
        result,
        (center_x, center_y),
        5,
        (0, 0, 255),
        -1
    )

    # Label candidate
    cv2.putText(
        result,
        f"Candidate {i + 1}",
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )


# --------------------------------
# DISPLAY RESULT
# --------------------------------

cv2.imshow(
    "Stage 3 - Human Candidate Detection",
    result
)

cv2.waitKey(0)
cv2.destroyAllWindows()