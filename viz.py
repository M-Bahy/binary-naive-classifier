import numpy as np
import matplotlib.pyplot as plt

# Sample multi-spectral pixel
pixel = np.array([5, 6, 4, 2, 9])

# # Method 1: Plot spectrum as a line graph
# plt.figure(figsize=(10, 4))
# plt.subplot(121)
# plt.plot(pixel, 'bo-')
# plt.title('Spectral Profile')
# plt.xlabel('Band Number')
# plt.ylabel('Intensity')
# plt.grid(True)

# # Method 2: Plot as a bar graph
# plt.subplot(122)
# plt.bar(range(len(pixel)), pixel)
# plt.title('Band Intensities')
# plt.xlabel('Band Number')
# plt.ylabel('Intensity')

# plt.tight_layout()
# plt.show()

# For a full image (assuming you have a multi-spectral image)
def visualize_bands(image):
    """
    Visualize each band of a multi-spectral image separately
    image shape should be (height, width, bands)
    """
    n_bands = image.shape[-1]
    fig, axes = plt.subplots(1, n_bands, figsize=(3*n_bands, 3))
    
    for i in range(n_bands):
        axes[i].imshow(image[:,:,i], cmap='gray')
        axes[i].set_title(f'Band {i+1}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

# Example usage for a full image:
# test_image = np.random.rand(100, 100, 5)  # 100x100 pixels, 5 bands
# visualize_bands(test_image)
def create_composite_visualizations(image):
    """
    Create multiple visualizations of multi-band image
    """
    plt.figure(figsize=(15, 5))
    
    # 1. Standard RGB (first 3 bands)
    plt.subplot(131)
    rgb = image[:,:,:3]  # First 3 bands
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
    plt.imshow(rgb)
    plt.title('First 3 bands (RGB)')
    plt.axis('off')
    
    # 2. Average of all bands
    plt.subplot(132)
    avg_image = np.mean(image, axis=2)
    plt.imshow(avg_image, cmap='gray')
    plt.title('Average of all bands')
    plt.axis('off')
    
    # 3. Custom combination (e.g., bands 0,2,4)
    plt.subplot(133)
    custom = np.stack([image[:,:,0], image[:,:,2], image[:,:,4]], axis=2)
    custom = (custom - custom.min()) / (custom.max() - custom.min())
    plt.imshow(custom)
    plt.title('Custom band combination (0,2,4)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()