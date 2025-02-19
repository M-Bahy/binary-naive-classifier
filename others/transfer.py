import os
import shutil

def copy_files(src_dir, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    copied_files = set()

    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file not in copied_files:
                src_file_path = os.path.join(root, file)
                dest_file_path = os.path.join(dest_dir, file)
                shutil.copy2(src_file_path, dest_file_path)
                copied_files.add(file)

if __name__ == "__main__":
    src_directory = "/home/bahy/Desktop/CMS/Deep Learning/naive-classifier/Dataset/300/BSDS300-human/BSDS300/human/color"
    dest_directory = "/home/bahy/Desktop/CMS/Deep Learning/naive-classifier/Dataset/bahy/labels"
    copy_files(src_directory, dest_directory)
    print(f"All files have been copied to {dest_directory}")