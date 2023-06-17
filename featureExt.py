import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import pandas as pd

def estimate_blur(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def estimate_brightness(image):
    return np.average(image)

def estimate_contrast(image):
    return np.std(image)

def estimate_sharpness(image):
    gy, gx = np.gradient(image)
    gnorm = np.sqrt(gx**2 + gy**2)
    sharpness = np.average(gnorm)
    return sharpness

def normalize(value, min_value, max_value):
    normalized_value = (value - min_value) / (max_value - min_value)
    return normalized_value

# Define directory with images
directory = '/content/XR_Print_Foff'

# Define (arbitrary) minimum and maximum values for the metrics
min_max_values = {
    'blur': (100, 3000),
    'brightness': (0, 255),
    'contrast': (0, 100),
    'sharpness': (0, 20)
}

# List to store results
results = []

# Read all images in directory
for image_path in glob.glob(os.path.join(directory, '*')):
    # Load image as grayscale
    image = cv2.imread(image_path, 0)

    blur = estimate_blur(image)
    brightness = estimate_brightness(image)
    contrast = estimate_contrast(image)
    sharpness = estimate_sharpness(image)

    # Normalize metrics
    blur_normalized = normalize(blur, min_max_values['blur'][0], min_max_values['blur'][1])
    brightness_normalized = normalize(brightness, min_max_values['brightness'][0], min_max_values['brightness'][1])
    contrast_normalized = normalize(contrast, min_max_values['contrast'][0], min_max_values['contrast'][1])
    sharpness_normalized = normalize(sharpness, min_max_values['sharpness'][0], min_max_values['sharpness'][1])

    # Add data to results list
    results.append({
        'image_path': image_path,
        'blur': blur_normalized,
        'brightness': brightness_normalized,
        'contrast': contrast_normalized,
        'sharpness': sharpness_normalized
    })

# Create DataFrame from results list
df = pd.DataFrame(results)

# Save DataFrame to CSV
df.to_csv('image_quality_metrics1.2.csv', index=False)
