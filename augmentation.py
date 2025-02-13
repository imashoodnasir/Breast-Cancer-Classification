import cv2
import numpy as np
import os
import random
from glob import glob

def rotate_image(image, angle_range=(-55, 55), step=5):
    """ Rotate image randomly within a given range """
    angle = random.randrange(angle_range[0], angle_range[1], step)
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

def flip_image(image):
    """ Randomly flip the image horizontally or vertically """
    flip_type = random.choice([-1, 0, 1])  # -1: both, 0: vertical, 1: horizontal
    return cv2.flip(image, flip_type)

def add_noise(image, mean=0, var=0.25):
    """ Add Gaussian noise to the image """
    row, col, ch = image.shape
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = np.clip(image + gauss * 255, 0, 255).astype(np.uint8)
    return noisy

def augment_image(image):
    """ Apply random augmentation: rotation, flipping, noise """
    if random.choice([True, False]):
        image = rotate_image(image)
    if random.choice([True, False]):
        image = flip_image(image)
    if random.choice([True, False]):
        image = add_noise(image)
    return image

def augment_dataset(input_folder, output_folder, num_augments=30):
    """ Augment all images in a dataset and save them """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_paths = glob(os.path.join(input_folder, "*.jpg"))  # Modify for other formats if needed
    for img_path in image_paths:
        image = cv2.imread(img_path)
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        for i in range(num_augments):
            augmented_image = augment_image(image)
            output_path = os.path.join(output_folder, f"{base_name}_aug_{i}.jpg")
            cv2.imwrite(output_path, augmented_image)

# Example usage
input_folder = "dataset/original"
output_folder = "dataset/augmented"
augment_dataset(input_folder, output_folder)
