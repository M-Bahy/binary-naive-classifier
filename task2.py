import os
from random import shuffle
from PIL import Image
import numpy as np


mode = 3
images_directory = f"/home/bahy/Desktop/CMS/Deep Learning/naive-classifier/Dataset/bahy/{mode}_images"
labels_directory = "/home/bahy/Desktop/CMS/Deep Learning/naive-classifier/Dataset/bahy/labels"


def process_images(image_files):
    all_pixels = []
    for image_file in image_files:
        path = os.path.join(images_directory, image_file)
        image = Image.open(path)
        image_array = np.array(image)
        if image_array.ndim == 2:  # Grayscale image
            image_array = image_array[:, :, np.newaxis]
        pixels = image_array.reshape(-1, image_array.shape[-1])
        all_pixels.append(pixels)
    return np.vstack(all_pixels)


def process_labels(label_files):
    all_labels = []
    for label_file in label_files:
        path = os.path.join(labels_directory, label_file)
        with open(path, 'r') as file:
            labels = np.array([int(value) for line in file for value in line.split()])
            all_labels.append(labels)
    return np.hstack(all_labels).reshape(-1, 1)


def train_test_split(images_directory, labels_directory, train_size = 0.8):
    images = [f for f in os.listdir(images_directory) if os.path.isfile(os.path.join(images_directory, f))]
    labels = [f for f in os.listdir(labels_directory) if os.path.isfile(os.path.join(labels_directory, f))]
    images.sort()
    labels.sort()
    dataset = zip(images, labels)
    dataset = list(dataset)
    shuffle(dataset)
    images, labels = zip(*dataset)
    train_size = int(len(dataset) * train_size)
    x_train, y_train = images[:train_size], labels[:train_size]
    x_test, y_test = images[train_size:], labels[train_size:]
    return process_images(x_train), process_labels(y_train), process_images(x_test), process_labels(y_test)

def BayesModel(data,truth):
    model = {}
    # count the number of zeros and 255s in the truth
    truth = truth.flatten()
    count = np.bincount(truth)
    zeros = count[0]
    ones = count[255]
    model["P(0)"] = zeros / len(truth)
    model["P(1)"] = ones / len(truth)
    # for each class calculate the mean and std of each feature (pixel) (can be rgb or grayscale)
    print(model)
    print(data[0])
    
    

def BayesPredict(model,test_data):
    pass

def ConfMtrx(actual,predicted):
    pass

if __name__ == "__main__":
    if mode not in [1, 3, 204]:
        raise ValueError("Mode should be 1,3 or 204")
    x_train, y_train, x_test, y_test = train_test_split(images_directory, labels_directory)
    BM = BayesModel(x_train, y_train)
    lbl = BayesPredict(BM, x_test)
    Mtrx = ConfMtrx(y_test, lbl)