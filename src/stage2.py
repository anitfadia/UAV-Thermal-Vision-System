import cv2

print("STAGE 2 STARTED")

# Load image
image = cv2.imread(
    "dataset/images/test.jpg",
    cv2.IMREAD_GRAYSCALE
)

if image is None:
    print("ERROR: Could not load image")
    exit()

print("Image loaded successfully!")


# Normalization
normalized = cv2.normalize(
    image,
    None,
    0,
    255,
    cv2.NORM_MINMAX
)

print("Normalization completed!")


# CLAHE
clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

clahe_image = clahe.apply(normalized)

print("CLAHE completed!")


# Bilateral filtering
filtered = cv2.bilateralFilter(
    clahe_image,
    9,
    75,
    75
)

print("Bilateral filtering completed!")


# Save images
cv2.imwrite(
    "output/normalized.jpg",
    normalized
)

cv2.imwrite(
    "output/clahe_stage2.jpg",
    clahe_image
)

cv2.imwrite(
    "output/bilateral_filtered.jpg",
    filtered
)

print("Images saved successfully!")


# Display
cv2.imshow("Original", image)
cv2.imshow("Normalized", normalized)
cv2.imshow("CLAHE", clahe_image)
cv2.imshow("Bilateral Filtered", filtered)

print("Four windows should now be open.")
print("Press any key inside an image window to close them.")

cv2.waitKey(0)
cv2.destroyAllWindows()

print("STAGE 2 FINISHED")