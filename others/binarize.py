import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def binarize_image(image, threshold=128):
    # Convert image to grayscale
    grayscale_image = image.convert('L')
    # Convert grayscale image to numpy array
    grayscale_array = np.array(grayscale_image)
    # Apply thresholding
    binarized_array = np.where(grayscale_array > threshold, 255, 0).astype(np.uint8)
    # Convert back to image
    binarized_image = Image.fromarray(binarized_array)
    return binarized_image

def draw_comparison(img_path):
    original_image = Image.open(img_path)
    binarized_image = binarize_image(original_image)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), facecolor='red')
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(binarized_image, cmap='gray')
    axes[1].set_title('Binarized Image')
    axes[1].axis('off')

    plt.show()

if __name__ == "__main__":
    number = 147021
    img_path = f"/home/bahy/Desktop/CMS/Deep Learning/naive-classifier/Dataset/bahy/images/{number}.jpg"
    draw_comparison(img_path)