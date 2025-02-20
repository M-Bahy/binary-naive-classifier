import os
from collections import defaultdict
import random

# Total number of datapoints: 6435
# Unique labels: {'1': 1533, '2': 703, '3': 1358, '4': 626, '5': 707, '7': 1508}
# test would be 1287 so an entire class could end up in the test set 
# unlike task2.py where we deal with images and we split whole images into train and test sets

def process_file(input_path):
    directory = os.path.dirname(input_path)
    
    # Create paths for output files
    x_train_path = os.path.join(directory, 'x_train.txt')
    y_train_path = os.path.join(directory, 'y_train.txt')
    x_test_path = os.path.join(directory, 'x_test.txt')
    y_test_path = os.path.join(directory, 'y_test.txt')
    
    # Store data by class
    class_data = defaultdict(list)
    line_count = 0
    
    # First pass: Read and organize data by class
    with open(input_path, 'r') as input_file:
        for line in input_file:
            line_count += 1
            numbers = line.strip().split()
            
            if len(numbers) != 37:
                print(f"Warning: Line {line_count} has {len(numbers)} numbers instead of 37")
                continue
            
            features = numbers[:-1]
            label = numbers[-1]
            class_data[label].append((features, label))
    
    # Initialize files
    with open(x_train_path, 'w') as x_train_file, \
         open(y_train_path, 'w') as y_train_file, \
         open(x_test_path, 'w') as x_test_file, \
         open(y_test_path, 'w') as y_test_file:
        
        # Process each class
        for label, data in class_data.items():
            # Shuffle data for this class
            random.shuffle(data)
            
            # Calculate split point (80%)
            split_point = int(len(data) * 0.8)
            
            # Train data (80%)
            for features, lbl in data[:split_point]:
                x_train_file.write(' '.join(features) + '\n')
                y_train_file.write(lbl + '\n')
            
            # Test data (20%)
            for features, lbl in data[split_point:]:
                x_test_file.write(' '.join(features) + '\n')
                y_test_file.write(lbl + '\n')
            
            print(f"Class {label}: Total={len(data)}, Train={split_point}, Test={len(data)-split_point}")
    
    print(f"\nTotal number of lines processed: {line_count}")
    print(f"Created train/test files in: {directory}")

if __name__ == "__main__":
    input_file_path = "/media/bahy/MEDO BAHY/CMS/Deep Learning/naive-classifier/Dataset/bahy/36_images/dataset.txt"
    process_file(input_file_path)