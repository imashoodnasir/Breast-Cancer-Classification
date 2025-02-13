import cv2
import numpy as np
from skimage import exposure
import matplotlib.pyplot as plt

def dark_channel(image, size=15):
    """
    Compute the dark channel of an image using a minimum filter.
    """
    min_channel = np.min(image, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dark_channel_img = cv2.erode(min_channel, kernel)
    return dark_channel_img

def atmospheric_light(image, dark_channel_img, top_percent=0.001):
    """
    Estimate the atmospheric light from the dark channel.
    """
    num_pixels = int(top_percent * dark_channel_img.size)
    dark_vec = dark_channel_img.ravel()
    indices = np.argsort(dark_vec)[-num_pixels:]
    
    bright_pixels = image.reshape(-1, 3)[indices]
    A = np.mean(bright_pixels, axis=0)  # Estimate atmospheric light
    return A

def haze_removal(image, omega=0.95, size=15):
    """
    Apply haze removal using dark channel prior and atmospheric light estimation.
    """
    image = image.astype(np.float32) / 255.0
    dark_channel_img = dark_channel(image, size)
    A = atmospheric_light(image, dark_channel_img)
    
    transmission = 1 - omega * (dark_channel_img / A.min())
    transmission = np.clip(transmission, 0.1, 1)

    dehazed = (image - A) / transmission[:, :, np.newaxis] + A
    dehazed = np.clip(dehazed, 0, 1)
    
    return (dehazed * 255).astype(np.uint8)

def adaptive_histogram_equalization(image):
    """
    Apply Adaptive Histogram Equalization (AHE) using Contrast Limited Adaptive Histogram Equalization (CLAHE).
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_eq = clahe.apply(l)

    lab_eq = cv2.merge((l_eq, a, b))
    enhanced_img = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    
    return enhanced_img

def hrat_pipeline(image_path):
    """
    Apply Haze-Removed Adaptive Technique (HRAT) to an image.
    """
    # Load image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (256, 256))  # Resize for consistency

    # Step 1: Haze Removal
    haze_free_image = haze_removal(image)

    # Step 2: Adaptive Histogram Equalization
    enhanced_image = adaptive_histogram_equalization(haze_free_image)

    return image, haze_free_image, enhanced_image

# Example Usage
image_path = "sample_mammogram.jpg"  # Replace with actual mammogram image path
original, haze_free, enhanced = hrat_pipeline(image_path)

# Plot the results
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(cv2.cvtColor(haze_free, cv2.COLOR_BGR2RGB))
axes[1].set_title("Haze-Free Image")
axes[1].axis("off")

axes[2].imshow(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
axes[2].set_title("Enhanced Image (HRAT)")
axes[2].axis("off")

plt.show()
