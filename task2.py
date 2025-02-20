import os
from random import shuffle , choice
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


mode = 1
test_image = ""
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
    labels = np.hstack(all_labels).reshape(-1, 1)
    labels.flatten()
    labels = np.where(labels == 255, 1, labels)
    return labels

def train_test_split(images_directory, labels_directory, train_size = 0.8):
    global test_image
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
    test_image = choice(x_test)
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

def BayesModel(data, truth):
    model = {}
    # count the number of zeros and 255s in the truth
    truth = truth.flatten()
    count = count_values(truth)
    zeros = count[0]
    # print ("Actual count of 0s: ", zeros)
    ones = count[1]
    # print ("Actual count of 255s: ", ones)
    model["P(0)"] = zeros / len(truth)
    model["P(1)"] = ones / len(truth)

    # Initialize lists for each class
    grayscale_0 = []
    grayscale_1 = []
    red_0 = []
    red_1 = []
    green_0 = []
    green_1 = []
    blue_0 = []
    blue_1 = []

    for i in range(len(data)):
        sample_point = data[i]
        point_class = truth[i]
        number_of_channels = len(sample_point)
        
        if number_of_channels == 1:
            if point_class == 0:
                grayscale_0.append(sample_point[0])
            else:
                grayscale_1.append(sample_point[0])
        elif number_of_channels == 3:
            if point_class == 0:
                red_0.append(sample_point[0])
                green_0.append(sample_point[1])
                blue_0.append(sample_point[2])
            else:
                red_1.append(sample_point[0])
                green_1.append(sample_point[1])
                blue_1.append(sample_point[2])

    if number_of_channels == 1:
        # Calculate statistics for each class in grayscale
        mean_0, var_0 = stats(grayscale_0)
        mean_1, var_1 = stats(grayscale_1)
        model["mean_0_grayscale"] = mean_0
        model["variance_0_grayscale"] = var_0
        model["mean_1_grayscale"] = mean_1
        model["variance_1_grayscale"] = var_1
    elif number_of_channels == 3:
        # Calculate statistics for each class in RGB
        red_mean_0, red_var_0 = stats(red_0)
        red_mean_1, red_var_1 = stats(red_1)
        model["mean_0_red"] = red_mean_0
        model["variance_0_red"] = red_var_0
        model["mean_1_red"] = red_mean_1
        model["variance_1_red"] = red_var_1

        green_mean_0, green_var_0 = stats(green_0)
        green_mean_1, green_var_1 = stats(green_1)
        model["mean_0_green"] = green_mean_0
        model["variance_0_green"] = green_var_0
        model["mean_1_green"] = green_mean_1
        model["variance_1_green"] = green_var_1

        blue_mean_0, blue_var_0 = stats(blue_0)
        blue_mean_1, blue_var_1 = stats(blue_1)
        model["mean_0_blue"] = blue_mean_0
        model["variance_0_blue"] = blue_var_0
        model["mean_1_blue"] = blue_mean_1
        model["variance_1_blue"] = blue_var_1

    return model

def BayesPredict(model, test_data):
    lbl = []
    for sample in test_data:
        if sample.shape[0] == 1:
            # Calculate for class 0
            p_gray_given_0 = (1 / ((2 * np.pi * model["variance_0_grayscale"]) ** 0.5)) * np.exp(-((sample[0] - model["mean_0_grayscale"]) ** 2) / (2 * model["variance_0_grayscale"]))
            log_p_gray_given_0 = np.log(p_gray_given_0)
            class_0_prediction = log_p_gray_given_0 + np.log(model["P(0)"])
            
            # Calculate for class 1
            p_gray_given_1 = (1 / ((2 * np.pi * model["variance_1_grayscale"]) ** 0.5)) * np.exp(-((sample[0] - model["mean_1_grayscale"]) ** 2) / (2 * model["variance_1_grayscale"]))
            log_p_gray_given_1 = np.log(p_gray_given_1)
            class_1_prediction = log_p_gray_given_1 + np.log(model["P(1)"])
            
            lbl.append(0 if class_0_prediction > class_1_prediction else 1)
        elif sample.shape[0] == 3:
            # Calculate for class 0
            p_red_given_0 = (1 / ((2 * np.pi * model["variance_0_red"]) ** 0.5)) * np.exp(-((sample[0] - model["mean_0_red"]) ** 2) / (2 * model["variance_0_red"]))
            p_green_given_0 = (1 / ((2 * np.pi * model["variance_0_green"]) ** 0.5)) * np.exp(-((sample[1] - model["mean_0_green"]) ** 2) / (2 * model["variance_0_green"]))
            p_blue_given_0 = (1 / ((2 * np.pi * model["variance_0_blue"]) ** 0.5)) * np.exp(-((sample[2] - model["mean_0_blue"]) ** 2) / (2 * model["variance_0_blue"]))
            
            log_p_red_given_0 = np.log(p_red_given_0)
            log_p_green_given_0 = np.log(p_green_given_0)
            log_p_blue_given_0 = np.log(p_blue_given_0)
            
            class_0_prediction = log_p_red_given_0 + log_p_green_given_0 + log_p_blue_given_0 + np.log(model["P(0)"])
            
            # Calculate for class 1
            p_red_given_1 = (1 / ((2 * np.pi * model["variance_1_red"]) ** 0.5)) * np.exp(-((sample[0] - model["mean_1_red"]) ** 2) / (2 * model["variance_1_red"]))
            p_green_given_1 = (1 / ((2 * np.pi * model["variance_1_green"]) ** 0.5)) * np.exp(-((sample[1] - model["mean_1_green"]) ** 2) / (2 * model["variance_1_green"]))
            p_blue_given_1 = (1 / ((2 * np.pi * model["variance_1_blue"]) ** 0.5)) * np.exp(-((sample[2] - model["mean_1_blue"]) ** 2) / (2 * model["variance_1_blue"]))
            
            log_p_red_given_1 = np.log(p_red_given_1)
            log_p_green_given_1 = np.log(p_green_given_1)
            log_p_blue_given_1 = np.log(p_blue_given_1)
            
            class_1_prediction = log_p_red_given_1 + log_p_green_given_1 + log_p_blue_given_1 + np.log(model["P(1)"])
            
            lbl.append(0 if class_0_prediction > class_1_prediction else 1)
    return lbl

def ConfMtrx(actual, predicted):
    # Initialize confusion matrix
    # [TN FP]
    # [FN TP]
    confusion_matrix = [[0, 0], 
                       [0, 0]]
    
    # Flatten actual array if it's 2D
    actual = actual.flatten()
    actual = np.where(actual == 255, 1, actual)
    
    # Fill confusion matrix
    for i in range(len(actual)):
        if actual[i] == 0 and predicted[i] == 0:
            confusion_matrix[0][0] += 1  # True Negative
        elif actual[i] == 0 and predicted[i] == 1:
            confusion_matrix[0][1] += 1  # False Positive
        elif actual[i] == 1 and predicted[i] == 0:
            confusion_matrix[1][0] += 1  # False Negative
        else:
            confusion_matrix[1][1] += 1  # True Positive
    
    # Calculate metrics
    tn, fp = confusion_matrix[0]
    fn, tp = confusion_matrix[1]
    
    # Avoid division by zero
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Print results
    print("\nConfusion Matrix:")
    print("-" * 21)
    print(f"|    TN: {tn:<8} FP: {fp:<8}|")
    print(f"|    FN: {fn:<8} TP: {tp:<8}|")
    print("-" * 21)
    print("\nMetrics:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1_score:.4f}")
    
    return confusion_matrix

def is_grayscale(image_array):
    # If 2D array, it's grayscale
    if len(image_array.shape) == 2:
        return True
    # If 3D array with single channel, it's grayscale
    elif len(image_array.shape) == 3:
        return image_array.shape[2] == 1
    return False

def visualize(model):
    # Get original image and create ground truth
    original_image = Image.open(os.path.join(images_directory, test_image))
    original_array = np.array(original_image)
    is_gray_scale = is_grayscale(original_array)
    # Create ground truth image (threshold at 128)
    grayscale_image = original_image.convert('L')
    ground_truth = np.array(grayscale_image)
    ground_truth = np.where(ground_truth > 128, 255, 0)
    
    # Create prediction image
    height, width = original_array.shape[:2]
    prediction = np.zeros((height, width), dtype=np.uint8)
    
    # Process each pixel
    for i in range(height):
        for j in range(width):
            if len(original_array.shape) == 3:  # RGB image
                pixel = original_array[i, j]
                test_pixel = np.array([[pixel[0], pixel[1], pixel[2]]])
            else:  # Grayscale image
                pixel = original_array[i, j]
                test_pixel = np.array([[pixel]])
                
            pred = BayesPredict(model, test_pixel)
            prediction[i, j] = 255 if pred[0] == 1 else 0
    
    # Create figure with three subplots
    
# Create figure with three subplots with red background
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('red')  # Set figure background to red
    
    # Plot original image
    if is_gray_scale:
        ax1.imshow(original_array, cmap='gray')
    else:
        ax1.imshow(original_array)
    ax1.set_title('Original Image', color='blue')  # Set title color to blue
    ax1.set_facecolor('red')  # Set subplot background to red
    ax1.axis('off')
    
    # Plot ground truth
    if is_gray_scale:
        ax2.imshow(ground_truth, cmap='gray')
    else:
        ax2.imshow(ground_truth)
    ax2.set_title('Ground Truth', color='blue')  # Set title color to blue
    ax2.set_facecolor('red')  # Set subplot background to red
    ax2.axis('off')
    
    # Plot prediction
    if is_gray_scale:
        ax3.imshow(prediction, cmap='gray')
    else:
        ax3.imshow(prediction)
    ax3.set_title('Prediction', color='blue')  # Set title color to blue
    ax3.set_facecolor('red')  # Set subplot background to red
    ax3.axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if mode not in [1, 3, 36]:
        raise ValueError("Mode should be 1,3 or 36")
    x_train, y_train, x_test, y_test = train_test_split(images_directory, labels_directory)
    BM = BayesModel(x_train, y_train)
    print(BM)
    lbl = BayesPredict(BM, x_test)
    Mtrx = ConfMtrx(y_test, lbl)
    visualize(BM)