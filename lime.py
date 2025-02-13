import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries

def predict_fn(images):
    """ Model prediction function for LIME """
    images = np.array(images) / 255.0
    return model.predict(images)

def apply_lime(image_path, model):
    """ Apply LIME on an image """
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        img.astype('double'), predict_fn, top_labels=1, hide_color=0, num_samples=1000
    )

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], positive_only=True, num_features=10, hide_rest=False
    )

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(mark_boundaries(temp, mask))
    plt.title("LIME Explanation")
    plt.axis("off")

    plt.show()

# Example Usage
apply_lime(image_path, model)
