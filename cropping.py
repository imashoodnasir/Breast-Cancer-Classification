import cv2
import numpy as np
import matplotlib.pyplot as plt

def crop_mammogram(image, margin=(20, 20, 40, 20)):
    """
    Crop the mammogram image to ensure uniformity.
    Removes unnecessary background while keeping breast tissue centered.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # Find contours of the breast region
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image  # Return original if no contours found

    # Get bounding box around the largest detected contour
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    # Apply margins
    x = max(0, x - margin[0])
    y = max(0, y - margin[2])
    w = min(image.shape[1] - x, w + margin[1])
    h = min(image.shape[0] - y, h + margin[3])

    cropped_img = image[y:y+h, x:x+w]
    
    return cropped_img

def normalize_image(image, target_size=(224, 224)):
    """
    Resize and normalize image.
    """
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32) / 255.0  # Normalize pixel values to [0,1]
    return image

def preprocess_mammogram(image_path):
    """
    Full preprocessing pipeline: Load, crop, normalize.
    """
    image = cv2.imread(image_path)
    cropped = crop_mammogram(image)
    normalized = normalize_image(cropped)

    return image, cropped, normalized

# Example usage
image_path = "sample_mammogram.jpg"  # Replace with actual image path
original, cropped, normalized = preprocess_mammogram(image_path)

# Display images
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
axes[1].set_title("Cropped Image")
axes[1].axis("off")

axes[2].imshow(normalized)
axes[2].set_title("Normalized Image")
axes[2].axis("off")

plt.show()
