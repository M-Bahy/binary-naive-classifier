import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def create_synthetic_multispectral_image(rgb_image, num_bands=204):
    # Convert RGB image to numpy array
    rgb_array = np.array(rgb_image)
    height, width, _ = rgb_array.shape
    
    # Initialize multi-spectral image with zeros
    multispectral_image = np.zeros((height, width, num_bands), dtype=np.float32)
    
    # Copy RGB channels to the first three bands
    multispectral_image[:, :, :3] = rgb_array / 255.0  # Normalize to [0, 1]
    
    # Generate synthetic bands by applying transformations
    for i in range(3, num_bands):
        if i % 3 == 0:
            multispectral_image[:, :, i] = np.mean(rgb_array, axis=2) / 255.0  # Grayscale for num_bands divisible by 3
        elif i % 3 == 1:
            multispectral_image[:, :, i] = np.std(rgb_array, axis=2) / 255.0  # Standard deviation for num_bands 
        else:
            multispectral_image[:, :, i] = np.max(rgb_array, axis=2) / 255.0  # Max value for num_bands 
    
    return multispectral_image

def draw_comparison(img_path, multispectral_image):
    original_image = Image.open(img_path)
    # For visualization, we'll use the first three bands of the multi-spectral image
    multispectral_rgb = multispectral_image[:, :, :3]
    multispectral_rgb = (multispectral_rgb * 255).astype(np.uint8)
    multispectral_rgb_image = Image.fromarray(multispectral_rgb)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), facecolor='red')
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(multispectral_rgb_image)
    axes[1].set_title('Synthetic Multi-Spectral Image (First 3 Bands)')
    axes[1].axis('off')

    plt.show()

if __name__ == "__main__":
    number = 147021
    img_path = f"/home/bahy/Desktop/CMS/Deep Learning/naive-classifier/Dataset/bahy/images/{number}.jpg"
    original_image = Image.open(img_path)
    multispectral_image = create_synthetic_multispectral_image(original_image)
    # print the shape of both images
    print(f"Original image shape (width, height): {original_image.size}")
    print(f"Synthetic multi-spectral image shape (height, width, channels): {multispectral_image.shape}")
