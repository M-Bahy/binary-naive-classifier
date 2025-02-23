import numpy as np
from viz import create_composite_visualizations

# Create a dummy multi-spectral image with 5 bands
height, width = 100, 100
n_bands = 5

# Generate random test image
test_image = np.random.rand(height, width, n_bands)

# Add some spatial patterns to make it more interesting
for band in range(n_bands):
    # Add gradients
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    X, Y = np.meshgrid(x, y)
    test_image[:,:,band] = (X + Y)/2
    
    # Add some random circles
    for _ in range(3):
        cx = np.random.randint(0, width)
        cy = np.random.randint(0, height)
        r = np.random.randint(5, 20)
        for i in range(height):
            for j in range(width):
                if (i-cy)**2 + (j-cx)**2 < r**2:
                    test_image[i,j,band] = 1.0

# Visualize the test image
create_composite_visualizations(test_image)