import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the image in grayscale
image_path = "sample_image.jpg"
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Flatten image to 1D array
pixels = img.flatten()

# Plot histogram
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.hist(pixels, bins=50, color="blue", alpha=0.7)
plt.title("Histogram of Pixel Intensities")

# Box Plot
plt.subplot(1, 3, 2)
sns.boxplot(x=pixels, color="red")
plt.title("Box Plot of Pixel Intensities")

# Scatter Plot (index vs pixel intensity)
plt.subplot(1, 3, 3)
plt.scatter(range(len(pixels)), pixels, alpha=0.5, s=1)
plt.title("Scatter Plot of Pixel Intensities")

plt.tight_layout()
plt.show()