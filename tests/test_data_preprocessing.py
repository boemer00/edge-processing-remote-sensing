import os
import cv2
import pytest

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'raw_data')

def test_image_readability():
    for folder in os.listdir(DATA_PATH):
        folder_path = os.path.join(DATA_PATH, folder)

        # Ensure it's a directory and not a stray file or another entity
        if os.path.isdir(folder_path):
            for image in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image)

                # Using OpenCV to read the image
                img = cv2.imread(image_path)

                # Assert that the image is readable (img should not be None)
                assert img is not None, f'Failed to read {image_path}'

def test_image_sizes_consistency():
    sizes = set()

    for folder in os.listdir(DATA_PATH):
        folder_path = os.path.join(DATA_PATH, folder)

        if os.path.isdir(folder_path):
            for image in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image)
                img = cv2.imread(image_path)

                assert img is not None, f'Failed to read {image_path}'

                # Storing the size (height, width, channels)
                sizes.add(img.shape)

    # Asserting that all images are of the same size
    assert len(sizes) == 1, f'Inconsistent image sizes found: {sizes}'
