import numpy as np
import matplotlib.pyplot as plt
import os

def get_image_specs(filename):
    """
    Determine image specifications from RAW file size
    Common Tetracam resolutions and band configurations
    """
    file_size = os.path.getsize(filename)
    bytes_per_pixel = 2  # Tetracam typically uses 16-bit (2 bytes) per pixel
    
    # Extended list of common Tetracam resolutions
    resolutions = [
        (2048, 1536),  # 3.1MP
        (1280, 1024),  # 1.3MP
        (1024, 768),   # 0.8MP
        (640, 480),    # VGA
    ]
    
    # Common band configurations
    possible_band_counts = [3, 4, 5, 6]
    
    print(f"File size: {file_size} bytes")
    
    for width, height in resolutions:
        total_pixels = width * height
        for bands in possible_band_counts:
            expected_size = total_pixels * bands * bytes_per_pixel
            if file_size == expected_size:
                return width, height, bands
            
    # If we get here, print debug information
    print("\nDebug information:")
    print("Attempted combinations:")
    for width, height in resolutions:
        total_pixels = width * height
        for bands in possible_band_counts:
            expected_size = total_pixels * bands * bytes_per_pixel
            print(f"Resolution {width}x{height}, {bands} bands -> Expected size: {expected_size} bytes")
    
    raise ValueError(f"Could not determine image specifications from file size {file_size}. Try adjusting bytes_per_pixel if your camera uses different bit depth.")
    
    for width, height in resolutions:
        total_pixels = width * height
        possible_bands = file_size / (total_pixels * bytes_per_pixel)
        if possible_bands.is_integer():
            return width, height, int(possible_bands)
    
    raise ValueError("Could not determine image specifications from file size")




def visualize_multispectral(image):
    """
    Visualize a multispectral image by averaging all bands
    
    Args:
        image (numpy.ndarray): 3D array of shape (height, width, bands)
    """
    # Get number of bands
    num_bands = image.shape[2]
    print(f"Number of spectral bands: {num_bands}")
    
    # Average all bands
    averaged_image = np.mean(image, axis=2)
    
    # Normalize to [0,1] range for visualization
    normalized = (averaged_image - averaged_image.min()) / (averaged_image.max() - averaged_image.min())
    
    # Display the image
    plt.figure(figsize=(10,10))
    plt.imshow(normalized, cmap='gray')
    plt.title('Averaged Multispectral Image')
    plt.colorbar(label='Normalized Intensity')
    plt.axis('off')
    plt.show()

def read_raw_image(filename, dtype=np.uint8):
    """
    Read a Tetracam RAW file with header
    """
    with open(filename, 'rb') as f:
        # Read and parse header
        header = f.read(16)
        print("Header bytes:", [hex(b) for b in header])
        
        # The actual width appears to be in bytes 4-5 (0x08, 0x10)
        width = 2048  # Based on common Tetracam resolution
        height = 1536  # Based on common Tetracam resolution
        bands = 3     # RGB format
        
        # Skip header
        raw_data = np.fromfile(f, dtype=dtype)
        print(f"Raw data size: {raw_data.size} bytes")
        
        # Calculate expected size
        expected_size = width * height * bands
        if raw_data.size != expected_size:
            print(f"Warning: Data size mismatch. Expected {expected_size}, got {raw_data.size}")
        
        try:
            # Reshape into 3D array (height, width, bands)
            # Note: Tetracam typically stores in band-interleaved format
            image = raw_data[:expected_size].reshape(height, width, bands)
            return image
        except ValueError as e:
            print(f"Raw data shape: {raw_data.shape}")
            print(f"Attempted reshape to: ({height}, {width}, {bands})")
            raise ValueError(f"Reshaping failed. Expected size: {expected_size}, available: {raw_data.size}")

if __name__ == "__main__":
    raw_file = "/home/bahy/Documents/Triton-Friday13 Test/TTC00003.RAW"
    try:
        # Try both 8-bit and 16-bit data types
        for dtype in [np.uint8, np.uint16]:
            try:
                print(f"\nTrying {dtype.__name__}...")
                image = read_raw_image(raw_file, dtype=dtype)
                print(f"Successfully read image with shape: {image.shape}")
                visualize_multispectral(image)
                break
            except Exception as e:
                print(f"Failed with {dtype.__name__}: {str(e)}")
    except Exception as e:
        print(f"Error: {str(e)}")