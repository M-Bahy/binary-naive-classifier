import os
from PIL import Image
import numpy as np
import rasterio
from rasterio.transform import Affine

def create_synthetic_multispectral_image(rgb_image, num_bands=204):
    # Convert RGB image to numpy array
    rgb_array = np.array(rgb_image)
    height, width, _ = rgb_array.shape
    
    # Initialize multi-spectral image with zeros
    multispectral_image = np.zeros((height, width, num_bands), dtype=np.float32)
    
    # Copy RGB channels to the first three bands
    multispectral_image[:, :, :3] = rgb_array / 255.0  # Normalize to [0, 1]
    
    # Generate synthetic bands using diverse transformations
    for i in range(3, num_bands):
        if i % 4 == 0:
            # Grayscale: Mean of RGB values
            multispectral_image[:, :, i] = np.mean(rgb_array, axis=2) / 255.0
        elif i % 4 == 1:
            # Edge detection: Sobel filter on grayscale
            from scipy.ndimage import sobel
            grayscale = np.mean(rgb_array, axis=2)
            multispectral_image[:, :, i] = sobel(grayscale) / 255.0
        elif i % 4 == 2:
            # Texture: Local binary patterns (simplified)
            grayscale = np.mean(rgb_array, axis=2)
            texture = np.zeros_like(grayscale)
            texture[1:-1, 1:-1] = (grayscale[1:-1, 1:-1] > grayscale[:-2, :-2]).astype(float)
            multispectral_image[:, :, i] = texture
        else:
            # Random noise to simulate spectral variations
            multispectral_image[:, :, i] = np.random.normal(0.5, 0.1, (height, width))
    
    return multispectral_image

def save_multispectral_image_as_tif(image_array, output_path):
    """
    Save a multi-spectral image (NumPy array) as a GeoTIFF file using rasterio.
    """
    height, width, num_bands = image_array.shape
    
    # Define the transform (identity matrix for simplicity)
    transform = Affine.identity()
    
    # Save as GeoTIFF
    with rasterio.open(
        output_path, 'w',
        driver='GTiff',
        height=height,
        width=width,
        count=num_bands,
        dtype=image_array.dtype,
        transform=transform
    ) as dst:
        for band in range(num_bands):
            dst.write(image_array[:, :, band], band + 1)

def process_images(original_dir, gray_scale_dir):
    if not os.path.exists(gray_scale_dir):
        os.makedirs(gray_scale_dir)
    # if not os.path.exists(multi_spectrum_dir):
    #     os.makedirs(multi_spectrum_dir)
    
    images = [f for f in os.listdir(original_dir) if os.path.isfile(os.path.join(original_dir, f))]
    
    for image_name in images:
        image_path = os.path.join(original_dir, image_name)
        image = Image.open(image_path)
        
        # Convert to grayscale
        gray_image = image.convert('L')
        gray_image.save(os.path.join(gray_scale_dir, image_name))
        
        # # Convert to multi-spectral
        # multispectral_image = create_synthetic_multispectral_image(image)
        
        # # Save the entire 204 channels as a .tif file
        # output_path = os.path.join(multi_spectrum_dir, image_name.replace('.jpg', '.tif'))
        # save_multispectral_image_as_tif(multispectral_image, output_path)

if __name__ == "__main__":
    original_images_dir = "/media/bahy/MEDO BAHY/CMS/Deep Learning/naive-classifier/Dataset/bahy/3_images"
    gray_scale_dir = "/media/bahy/MEDO BAHY/CMS/Deep Learning/naive-classifier/Dataset/bahy/1_images"
    # multi_spectrum_dir = "/media/bahy/MEDO BAHY/CMS/Deep Learning/naive-classifier/Dataset/bahy/204_images"
    
    process_images(original_images_dir, gray_scale_dir)