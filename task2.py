import os
from random import shuffle
from PIL import Image
import numpy as np


mode = 1
images_directory = f"/home/bahy/Desktop/CMS/Deep Learning/naive-classifier/Dataset/subset/{mode}_images"
labels_directory = "/home/bahy/Desktop/CMS/Deep Learning/naive-classifier/Dataset/subset/labels"


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

def count_values(array):
    # Count occurrences of each value
    max_value = max(array)
    counts = [0] * (max_value + 1)
    for value in array:
        counts[value] += 1
    return counts

def stats(values):
    """
    Calculate mean and standard deviation without using numpy methods
    Args:
        values: Array of values (either scalars or 1D arrays)
    Returns:
        tuple: (mean, std)
    """
    # Calculate mean
    sum_values = 0
    n = len(values)
    for value in values:
        sum_values += value
    mean = sum_values / n
    
    # Calculate variance
    sum_squared_diff = 0
    for value in values:
        # Check if value is array or scalar
        diff = value - mean
        sum_squared_diff += diff * diff
    variance = sum_squared_diff / n
    
    # # Calculate standard deviation
    # std = (variance) ** 0.5
    
    return mean, variance

def BayesModel(data,truth):
    model = {}
    # count the number of zeros and 255s in the truth
    truth = truth.flatten()
    count = count_values(truth)
    zeros = count[0]
    ones = count[255]
    model["P(0)"] = zeros / len(truth)
    model["P(1)"] = ones / len(truth)

    grayscale = []
    red = []
    green = []
    blue = []

    for i in range(len(data)):
        sample_point = data[i]
        point_class = truth[i]
        number_of_channels = len(sample_point)
        if number_of_channels == 1:
            grayscale.append(sample_point[0])
        elif number_of_channels == 3:
            red.append(sample_point[0])
            green.append(sample_point[1])
            blue.append(sample_point[2])
    if number_of_channels == 1:
        # calculate the mean and variance for grayscale samples , this is an array of n numbers
        grayscale_mean, grayscale_std = stats(grayscale)
        model["mean_grayscale"] = grayscale_mean
        model["variance_grayscale"] = grayscale_std
    elif number_of_channels == 3:
        # calculate the mean and variance for red samples , this is an array of n numbers
        red_mean, red_std = stats(red)
        model["mean_red"] = red_mean
        model["variance_red"] = red_std
        # calculate the mean and variance for green samples , this is an array of n numbers
        green_mean, green_std = stats(green)
        model["mean_green"] = green_mean
        model["variance_green"] = green_std
        # calculate the mean and variance for blue samples , this is an array of n numbers
        blue_mean, blue_std = stats(blue)
        model["mean_blue"] = blue_mean
        model["variance_blue"] = blue_std
    return model

        
    
    
    

def BayesPredict(model,test_data):
    lbl = []
    for sample in test_data :
        if sample.shape[0] == 1:
            # calculate the probability of the sample given the class 0
            p_gray_given_0 = (1 / ((2 * np.pi * model["variance_grayscale"]) ** 0.5)) * np.exp(-((sample[0] - model["mean_grayscale"]) ** 2) / (2 * model["variance_grayscale"]))
            log_p_gray_given_0 = np.log(p_gray_given_0)
            class_0_prediction = log_p_gray_given_0 + np.log(model["P(0)"])
            # calculate the probability of the sample given the class 1
            p_x_given_1 = (1 / ((2 * np.pi * model["variance_grayscale"]) ** 0.5)) * np.exp(-((sample[0] - model["mean_grayscale"]) ** 2) / (2 * model["variance_grayscale"]))
            log_p_gray_given_1 = np.log(p_x_given_1)
            class_1_prediction = log_p_gray_given_1 + np.log(model["P(1)"])
            # append the label to the list
            lbl.append(0 if class_0_prediction > class_1_prediction else 1)
        elif sample.shape[0] == 3:
            # calculate the probability of the sample given the class 0
            p_red_given_0 = (1 / ((2 * np.pi * model["variance_red"]) ** 0.5)) * np.exp(-((sample[0] - model["mean_red"]) ** 2) / (2 * model["variance_red"]) )
            log_p_red_given_0 = np.log(p_red_given_0)

            p_green_given_0 = (1 / ((2 * np.pi * model["variance_green"]) ** 0.5)) * np.exp(-((sample[1] - model["mean_green"]) ** 2) / (2 * model["variance_green"]) )
            log_p_green_given_0 = np.log(p_green_given_0)

            p_blue_given_0 = (1 / ((2 * np.pi * model["variance_blue"]) ** 0.5)) * np.exp(-((sample[2] - model["mean_blue"]) ** 2) / (2 * model["variance_blue"]))
            log_p_blue_given_0 = np.log(p_blue_given_0)

            class_0_prediction = log_p_red_given_0 + log_p_green_given_0 + log_p_blue_given_0 + np.log(model["P(0)"])
            # calculate the probability of the sample given the class 1
            p_red_given_1 = (1 / ((2 * np.pi * model["variance_red"]) ** 0.5)) * np.exp(-((sample[0] - model["mean_red"]) ** 2) / (2 * model["variance_red"]) )
            log_p_red_given_1 = np.log(p_red_given_1)
            
            p_green_given_1 = (1 / ((2 * np.pi * model["variance_green"]) ** 0.5)) * np.exp(-((sample[1] - model["mean_green"]) ** 2) / (2 * model["variance_green"]))
            log_p_green_given_1 = np.log(p_green_given_1)
            
            p_blue_given_1 = (1 / ((2 * np.pi * model["variance_blue"]) ** 0.5)) * np.exp(-((sample[2] - model["mean_blue"]) ** 2) / (2 * model["variance_blue"]))
            log_p_blue_given_1 = np.log(p_blue_given_1)
            
            class_1_prediction = log_p_red_given_1 + log_p_green_given_1 + log_p_blue_given_1 + np.log(model["P(1)"])
            
            # append the label to the list
            lbl.append(0 if class_0_prediction > class_1_prediction else 1)
    return lbl

def ConfMtrx(actual,predicted):
    pass

if __name__ == "__main__":
    if mode not in [1, 3, 204]:
        raise ValueError("Mode should be 1,3 or 204")
    x_train, y_train, x_test, y_test = train_test_split(images_directory, labels_directory)
    BM = BayesModel(x_train, y_train)
    # print(BM)
    lbl = BayesPredict(BM, x_test)
    # Mtrx = ConfMtrx(y_test, lbl)