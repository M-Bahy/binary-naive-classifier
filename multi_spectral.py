import os

def process_file(input_path):
    # Get the directory of the input file
    directory = os.path.dirname(input_path)
    
    # Create paths for output files
    images_path = os.path.join(directory, 'images.txt')
    labels_path = os.path.join(directory, 'labels.txt')
    
    # Initialize line counter
    line_count = 0
    unique = {
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0,
        "5": 0,
        "7": 0,
    }
    # Open all files
    with open(input_path, 'r') as input_file, \
         open(images_path, 'w') as images_file, \
         open(labels_path, 'w') as labels_file:
        
        # Process each line
        for line in input_file:
            line_count += 1
            # Split the line into numbers
            numbers = line.strip().split()
            
            # Verify the format (37 numbers per line)
            if len(numbers) != 37:
                print(f"Warning: Line {line_count} has {len(numbers)} numbers instead of 37")
                continue
            
            # Extract features (first 36 numbers) and label (last number)
            features = numbers[:-1]
            label = numbers[-1]
            unique[label] += 1
            # Write to respective files
            images_file.write(' '.join(features) + '\n')
            labels_file.write(label + '\n')
    
    print(f"Total number of lines processed: {line_count}")
    print(f"Created images file at: {images_path}")
    print(f"Created labels file at: {labels_path}")
    print(f"Unique labels: {unique}")

# Example usage
if __name__ == "__main__":
    # Replace with your input file path
    input_file_path = "/media/bahy/MEDO BAHY/CMS/Deep Learning/naive-classifier/Dataset/bahy/36_images/dataset.txt"
    process_file(input_file_path)