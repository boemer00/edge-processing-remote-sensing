import os
import cv2
import shutil
import numpy as np
import pandas as pd
from src.data.preprocessing import (
    generate_data_list,
    generate_dataframe_from_directory,
    resize_image,
    normalize_image
)

# Paths to the data directories relative to the test script location
RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'raw_data')
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'test_data')
BACKUP_DIR = os.path.join(os.path.dirname(__file__), 'backup_data')

# Utility functions for the tests
def create_backup(source_dir, backup_dir):
    # Create a backup of the source_dir in the backup_dir
    if os.path.exists(backup_dir):
        shutil.rmtree(backup_dir)
    shutil.copytree(source_dir, backup_dir)

def restore_from_backup(backup_dir, source_dir):
    # Restore the source_dir from the backup_dir
    if os.path.exists(source_dir):
        shutil.rmtree(source_dir)
    shutil.copytree(backup_dir, source_dir)

def is_image_file(filename):
    return any(filename.endswith(ext) for ext in ['.png', '.jpg', '.jpeg'])

def get_all_image_paths(directory):
    image_paths = []
    for subdir, dirs, files in os.walk(directory):
        for file in files:
            if is_image_file(file):
                image_paths.append(os.path.join(subdir, file))
    return image_paths

# Test 1: Data List Generation
def test_data_list_generation():
    data_list = generate_data_list(TEST_DATA_DIR)
    assert isinstance(data_list, list), "Data list should be a list."
    assert len(data_list) > 0, "Data list should not be empty."
    assert all('image_path' in item and 'label' in item for item in data_list), "Each item should contain 'image_path' and 'label'."

# Test 2: Data Frame Generation
def test_data_frame_generation():
    data_frame = generate_dataframe_from_directory(TEST_DATA_DIR)
    assert isinstance(data_frame, pd.DataFrame), "Should create a DataFrame."
    assert not data_frame.empty, "DataFrame should not be empty."
    assert 'image_path' in data_frame.columns and 'label' in data_frame.columns, "DataFrame should have 'image_path' and 'label' columns."

# Test 3: Resize Images
def test_resize_images():
    target_size = (100, 100)
    # Get all image paths
    image_paths = get_all_image_paths(TEST_DATA_DIR)
    for image_path in image_paths:
        # Read the image from file
        img = cv2.imread(image_path)
        # Check if the image was read correctly
        assert img is not None, f"Image at {image_path} could not be read."
        # Resize the image using the provided function from preprocessing module
        resized_img = resize_image(img, target_size)
        # Save the resized image back to file
        cv2.imwrite(image_path, resized_img)
        # Check if the image was resized correctly
        assert resized_img.shape[:2] == target_size, f"Image at {image_path} not resized correctly: {resized_img.shape[:2]} != {target_size}"

# Test 4: Normalize Images
def test_normalize_images():
    image_paths = get_all_image_paths(TEST_DATA_DIR)

    for image_path in image_paths:
        img = cv2.imread(image_path)
        assert img is not None, f"Image at {image_path} could not be read."

        # Assume normalize_image function modifies the image array directly
        img_normalized = normalize_image(img.copy())  # If normalize_image returns a new image, otherwise just use img directly

        # Convert to float32 for checking normalized values
        assert img_normalized.dtype == np.float32, "Normalized image should be float32 type"

        # Normalized images should have values close to the range [0, 1]
        assert np.all((img_normalized >= 0) & (img_normalized <= 1)), f"Image at {image_path} not normalized correctly"

if __name__ == "__main__":
    create_backup(TEST_DATA_DIR, BACKUP_DIR)  # Backup once at the start
    try:
        test_data_list_generation()
        test_data_frame_generation()
        test_resize_images()
        test_normalize_images()
    finally:
        restore_from_backup(BACKUP_DIR, TEST_DATA_DIR)  # Restore once after all tests
    print("All tests passed.")
