# File: tests/test_data_processing.py
import os
import numpy as np
import pandas as pd
import cv2
import shutil
from src.data.augmentation import augment_and_save_images, ImageAugmenter
from src.data.preprocessing import (
    generate_data_list,
    generate_dataframe_from_directory,
    resize_image,
    normalize_image
)


# Path to the raw data directory relative to the test script location
RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'raw_data')
BACKUP_DIR = 'To be created...'


def create_backup(source_dir, backup_dir):
    if not os.path.exists(backup_dir):
        os.makedirs(backup_dir)
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            src_path = os.path.join(root, file)
            dst_path = os.path.join(backup_dir, os.path.relpath(src_path, source_dir))
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            shutil.copy2(src_path, dst_path)

def restore_from_backup(backup_dir, source_dir):
    for root, dirs, files in os.walk(backup_dir):
        for file in files:
            src_path = os.path.join(root, file)
            dst_path = os.path.join(source_dir, os.path.relpath(src_path, backup_dir))
            shutil.copy2(src_path, dst_path)

# Test 1: Data List Generation
def test_data_list_generation():
    data_list = generate_data_list(RAW_DATA_DIR)
    assert isinstance(data_list, list)
    assert len(data_list) > 0  # Assumes there is at least one image
    assert all('image_path' in item and 'label' in item for item in data_list)

# Test 2: Data Frame Generation
def test_data_frame_generation():
    data_frame = generate_dataframe_from_directory(RAW_DATA_DIR)
    assert isinstance(data_frame, pd.DataFrame)
    assert not data_frame.empty
    assert 'image_path' in data_frame.columns and 'label' in data_frame.columns

# Test 3: Augmentation Process
def test_augmentation_process():
    augmenter = ImageAugmenter(angle=15, x_shift=10, y_shift=10)
    augment_and_save_images(RAW_DATA_DIR, augmenter)
    # This test assumes augmentation has been run prior and checks for the presence of augmented images
    assert count_specific_augmented_images(RAW_DATA_DIR, ['rotated', 'translated']) == 371

# Test 4: File Paths
def test_file_paths():
    data_frame = generate_dataframe_from_directory(RAW_DATA_DIR)
    all_files_exist = all(os.path.isfile(path) for path in data_frame['image_path'])
    assert all_files_exist

# Test 5: Augmentation Integrity
def test_augmentation_integrity():
    augmented_image_paths = get_augmented_image_paths(RAW_DATA_DIR)
    all_images_valid = all(is_valid_image(path) for path in augmented_image_paths)
    assert all_images_valid

# Test 6: Resize Images
def test_resize_images():
    create_backup(RAW_DATA_DIR, BACKUP_DIR)  # Backup images before resizing
    target_size = (100, 100)  # Assuming a smaller size for testing purposes
    resize_image(RAW_DATA_DIR, target_size)

    for root, _, files in os.walk(RAW_DATA_DIR):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                img = cv2.imread(image_path)
                assert img.shape[:2] == target_size, f"Image at {image_path} not resized correctly"

    restore_from_backup(BACKUP_DIR, RAW_DATA_DIR)  # Restore images from backup after test

# Test 7: Normalize Images
def test_normalize_images():
    create_backup(RAW_DATA_DIR, BACKUP_DIR)  # Backup images before normalization
    normalize_image(RAW_DATA_DIR)

    for root, _, files in os.walk(RAW_DATA_DIR):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                img = cv2.imread(image_path)
                img_normalized = img.astype(np.float32) / 255.0
                assert np.allclose(img, img_normalized * 255), f"Image at {image_path} not normalized correctly"

    restore_from_backup(BACKUP_DIR, RAW_DATA_DIR)  # Restore images from backup after test

# Utility function to check if images are valid
def is_valid_image(file_path):
    try:
        image = cv2.imread(file_path)
        return image is not None
    except Exception:
        return False

# Utility function to count augmented images with specific keywords
def count_specific_augmented_images(directory_path, keywords):
    augmented_count = 0
    for subdir, dirs, files in os.walk(directory_path):
        for file in files:
            if any(keyword in file for keyword in keywords):
                augmented_count += 1
    return augmented_count

# Utility function to retrieve paths of augmented images
def get_augmented_image_paths(directory_path):
    augmented_image_paths = []
    for subdir, dirs, files in os.walk(directory_path):
        for file in files:
            if any(keyword in file for keyword in ['rotated', 'translated']):
                augmented_image_paths.append(os.path.join(subdir, file))
    return augmented_image_paths
