import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
import cv2 
 
# Step 1: Load the MNIST dataset 
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data() 
 
# Step 2: Convert to 8-bit for Histogram Equalization 
x_train_uint8 = (x_train * 255).astype(np.uint8)  # Convert to uint8 for OpenCV 
x_test_uint8 = (x_test * 255).astype(np.uint8) 
 
# Step 3: Show Original Image 
def show_image(index): 
    plt.imshow(x_train[index], cmap="gray") 
    plt.title(f"Original Image - Label: {y_train[index]}") 
    plt.axis("off") 
    plt.show() 
 
show_image(69)  # Show the first image 
 
# Step 4: Apply Histogram Equalization 
def histogram_equalization(index): 
    original_image = x_train_uint8[index]  # Get original 8-bit image 
    equalized_image = cv2.equalizeHist(original_image)  # Apply histogram equalization 
 
    # Normalize the equalized image to [0, 1] range 
    normalized_image = equalized_image.astype(np.float32) / 255.0 
 
    # Display images side by side 
    fig, ax = plt.subplots(1, 3, figsize=(15, 5)) 
 
    ax[0].imshow(original_image, cmap="gray") 
    ax[0].set_title("Original Image") 
    ax[0].axis("off") 
 
    ax[1].imshow(equalized_image, cmap="gray") 
    ax[1].set_title("Histogram Equalized Image") 
    ax[1].axis("off") 
 
    ax[2].imshow(normalized_image, cmap="gray") 
    ax[2].set_title("Normalized Image (0-1)") 
    ax[2].axis("off") 
    plt.show() 
    return original_image, equalized_image, normalized_image 

original, equalized, normalized = histogram_equalization(69) 
# Step 5: Plot Histogram Before and After Equalization 
def plot_histogram(image, title): 
    plt.hist(image.flatten(), bins=50, color="blue", alpha=0.7, edgecolor="black") 
    plt.title(title) 
    plt.xlabel("Pixel Intensity") 
    plt.ylabel("Frequency") 
    plt.show() 
plot_histogram(original, "Original Image Histogram") 
plot_histogram(equalized, "Equalized Image Histogram") 
plot_histogram(normalized, "Normalized Image Histogram (0-1)")