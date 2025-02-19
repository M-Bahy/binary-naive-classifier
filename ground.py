import os
from PIL import Image
import numpy as np

def generate_ground_truth(images_directory, labels_directory , threshold = 128):
    """
    Generate the ground truth (binary class) for the images by a simple thresholding 
    Args:
        images_directory: directory containing the images
        labels_directory: directory to save the ground truth images
        threshold: threshold to binarize the images
    Returns:
        None
    """
    # loop through the images , convert them to gray scale and binarize (threshold = 128)
    images = [f for f in os.listdir(images_directory) if os.path.isfile(os.path.join(images_directory, f))]
    for image in images:
        image_path = os.path.join(images_directory, image)
        original_image = Image.open(image_path)
        gray_image = original_image.convert('L')
        binarized_image = np.where(np.array(gray_image) > threshold, 255, 0).astype(np.uint8)
        np.savetxt(os.path.join(labels_directory, f"{image.split('.')[0]}.txt"), binarized_image, fmt='%d')

if __name__ == "__main__":
    images_directory = "/home/bahy/Desktop/CMS/Deep Learning/naive-classifier/Dataset/bahy/images"
    labels_directory = "/home/bahy/Desktop/CMS/Deep Learning/naive-classifier/Dataset/bahy/labels"
    generate_ground_truth(images_directory, labels_directory)
    print("Ground truth generated successfully !")