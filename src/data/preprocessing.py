import os

import cv2
import numpy as np
import pandas as pd

# Given this script is located at "src/data/preprocessing.py"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_DATA_DIR = os.path.join(BASE_DIR, "raw_data")


def list_files_from_directory(directory_path):
    """Return a list of files from the given directory."""
    return [
        os.path.join(directory_path, f)
        for f in os.listdir(directory_path)
        if os.path.isfile(os.path.join(directory_path, f))
    ]


def get_data_from_directory(data_path):
    """Generate data list from a directory containing subdirectories of images."""
    data_list = []

    for folder in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder)

        if os.path.isdir(folder_path):
            for image_path in list_files_from_directory(folder_path):
                label = folder
                data_list.append({"image_path": image_path, "label": label})

    return data_list


def generate_dataframe_from_directory(data_path=RAW_DATA_DIR):
    """Generate a DataFrame from a directory containing subdirectories of images."""
    data_list = get_data_from_directory(data_path)
    return pd.DataFrame(data_list)


def get_image_paths_from_directory(directory_path):
    """Return a list of image paths from the given directory."""
    image_extensions = [".jpg", ".jpeg", ".png"]
    return [
        os.path.join(directory_path, f)
        for f in os.listdir(directory_path)
        if f.endswith(tuple(image_extensions))
    ]


def generate_data_list(data_directory=RAW_DATA_DIR):
    """Generate a list of image paths and their associated labels."""
    data_list = []
    for folder in os.listdir(data_directory):
        folder_path = os.path.join(data_directory, folder)
        if os.path.isdir(folder_path):
            for image_path in get_image_paths_from_directory(folder_path):
                label = folder
                data_list.append({"image_path": image_path, "label": label})
    return data_list


def resize_image(image, target_size=(224, 224)):
    """Resize a single image to the target size."""
    return cv2.resize(image, target_size)


def normalize_image(image):
    """Normalise a single image pixel values to the [0, 1] range."""
    return image.astype(np.float32) / 255.0


def process_images_in_directory(directory_path, target_size=(224, 224)):
    """Resize and normalise images in the given directory and its sub-directories."""
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                image_path = os.path.join(root, file)
                img = cv2.imread(image_path)
                if img is not None:
                    resized_img = resize_image(img, target_size)
                    normalized_img = normalize_image(resized_img)
                    cv2.imwrite(image_path, normalized_img * 255)


if __name__ == "__main__":
    data_df = generate_dataframe_from_directory()

    # Process images for resizing and normalisation
    process_images_in_directory(RAW_DATA_DIR)

    print(data_df.head())
