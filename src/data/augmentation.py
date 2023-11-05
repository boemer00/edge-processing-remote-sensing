import cv2
import os
import random
import numpy as np
from src.data.preprocessing import get_image_paths_from_directory

class ImageAugmenter:
    def __init__(self, angle=0, x_shift=0, y_shift=0):
        self.angle = angle
        self.x_shift = x_shift
        self.y_shift = y_shift

    def rotate_image(self, image, angle=None):
        if angle is None:
            angle = random.uniform(-self.angle, self.angle)
        rows, cols, _ = image.shape
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        return cv2.warpAffine(image, M, (cols, rows))

    def translate_image(self, image, x_shift=None, y_shift=None):
        if x_shift is None:
            x_shift = random.uniform(-self.x_shift, self.x_shift)
        if y_shift is None:
            y_shift = random.uniform(-self.y_shift, self.y_shift)
        rows, cols, _ = image.shape
        M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
        return cv2.warpAffine(image, M, (cols, rows))

# For pre-deployment data preparation
def augment_and_save_images(input_directory_path, output_directory_path, augmenter):
    image_paths = get_image_paths_from_directory(input_directory_path)
    random.shuffle(image_paths)

    for path in image_paths:
        image = cv2.imread(path)
        if image is None:
            print(f"Warning: Could not read image {path}. Skipping.")
            continue

        filename = os.path.basename(path)
        base_filename, file_extension = os.path.splitext(filename)

        # Apply different augmentations and save the new images
        rotated_image = augmenter.rotate_image(image)
        translated_image = augmenter.translate_image(image)

        # Save to the output directory with indicative filenames
        cv2.imwrite(os.path.join(output_directory_path, f"{base_filename}_rotated_{file_extension}"), rotated_image)
        cv2.imwrite(os.path.join(output_directory_path, f"{base_filename}_translated_{file_extension}"), translated_image)


# For run-time augmentation in the inference pipeline
def augment_image_for_inference(image, augmenter):
    return augmenter.rotate_image(image)

if __name__ == '__main__':
    # Pre-deployment preparation
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    RAW_DATA_DIR = os.path.join(BASE_DIR, 'raw_data')
    augmenter = ImageAugmenter(angle=15, x_shift=25, y_shift=25)
    for class_dir in os.listdir(RAW_DATA_DIR):
        class_path = os.path.join(RAW_DATA_DIR, class_dir)
        if os.path.isdir(class_path):
            augment_and_save_images(class_path, augmenter)
