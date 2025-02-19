import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def read_segmentation_file(seg_path):
    with open(seg_path, 'r') as file:
        lines = file.readlines()

    width = int(lines[4].split()[1])
    height = int(lines[5].split()[1])
    data_start_index = lines.index('data\n') + 1

    segmentation = np.zeros((height, width), dtype=np.uint8)

    for line in lines[data_start_index:]:
        class_id, row, start_col, end_col = map(int, line.split())
        segmentation[row, start_col:end_col + 1] = class_id

    return segmentation

def binarize_segmentation(segmentation):
    print("Segmentation array before binarization:")
    print(segmentation)
    
    binarized = np.where(segmentation == 0, 0, 1).astype(np.uint8)
    
    print("Binarized segmentation array:")
    print(binarized)
    
    return binarized

def draw_comparison(img_path, binarized_segmentation, original_segmentation=None):
    original_image = Image.open(img_path)  # Remove .convert('L') to keep original color
    binarized_image = Image.fromarray(binarized_segmentation * 255)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor='gray')
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    if original_segmentation is not None:
        axes[1].imshow(original_segmentation, cmap='gray')
        axes[1].set_title('Original Segmentation')
        axes[1].axis('off')

    axes[2].imshow(binarized_image, cmap='gray')
    axes[2].set_title('Binarized Segmentation')
    axes[2].axis('off')

    plt.show()

if __name__ == "__main__":
    number = 147021
    img_path = f"/home/bahy/Desktop/CMS/Deep Learning/naive-classifier/Dataset/bahy/images/{number}.jpg"
    original_image = Image.open(img_path)
    gray_image = original_image.convert('L')
    # in the gray_image, the pixel values are in the range [0, 255] , set the threshold to 128 to binarize the image
    threshold = 128
    binarized_image = np.where(np.array(gray_image) > threshold, 255, 0).astype(np.uint8)
    binarized_image = Image.fromarray(binarized_image)
    # draw a comparison between the original image and the binarized image
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), facecolor='red')
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    axes[1].imshow(binarized_image, cmap='gray')
    axes[1].set_title('Binarized Image')
    axes[1].axis('off')
    plt.show()