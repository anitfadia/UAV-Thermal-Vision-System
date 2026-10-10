import cv2
import os

input_path = "dataset/images/test.jpg"
output_folder = "output"

os.makedirs(output_folder, exist_ok=True)

image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print("ERROR: Could not load image.")
    print("Check the image path.")
    exit()

print("Image loaded successfully!")

# Create CLAHE
clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

# Apply CLAHE
enhanced = clahe.apply(image)

# Save enhanced image
output_path = os.path.join(
    output_folder,
    "clahe_output.jpg"
)

cv2.imwrite(output_path, enhanced)

print("CLAHE applied successfully!")
print("Output saved at:", output_path)

# Display original image
cv2.imshow("Original Thermal Image", image)

# Display CLAHE image
cv2.imshow("CLAHE Enhanced Image", enhanced)

# Wait until a key is pressed
cv2.waitKey(0)

# Close windows
cv2.destroyAllWindows()