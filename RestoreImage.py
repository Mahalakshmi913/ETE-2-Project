import cv2
import numpy as np

# Load images
OriginalImage = cv2.imread("damaged_image.jpg")
maskedDamages = cv2.imread("masked_image.jpg", cv2.IMREAD_GRAYSCALE)

# Check if images loaded correctly
if OriginalImage is None:
    print("Error: Damaged image not loaded properly.")
if maskedDamages is None:
    print("Error: Masked image not loaded properly.")

# Resize mask to match the original image size
mask = cv2.resize(maskedDamages, (OriginalImage.shape[1], OriginalImage.shape[0]))

# Convert mask to binary
_, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)

# Apply dilation to refine the mask
kernel = np.ones((7,7), np.uint8)
mask = cv2.dilate(mask, kernel, iterations=1)

# Inpaint the image
restoredImage = cv2.inpaint(OriginalImage, mask, 3, cv2.INPAINT_TELEA)

# Show results
cv2.imshow("Damaged Image", OriginalImage)
cv2.imshow("Masked Damages", maskedDamages)
cv2.imshow("Final Mask", mask)
cv2.imshow("Restored Image", restoredImage)

cv2.waitKey(0)
cv2.destroyAllWindows()
