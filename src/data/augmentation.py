import cv2
import os
import random
import numpy as np
from data.augmentation import get_image_paths_from_directory

class ImageAugmenter:
    def __init__(self, angle=0, x_shift=0, y_shift=0):
        self.angle = angle
        self.x_shift = x_shift
        self.y_shift = y_shift

    def rotate_image(self, image, angle=None):
        if angle is None:
            angle = self.angle
        rows, cols, _ = image.shape
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        return cv2.warpAffine(image, M, (cols, rows))

    def translate_image(self, image, x_shift=None, y_shift=None):
        if x_shift is None:
            x_shift = self.x_shift
        if y_shift is None:
            y_shift = self.y_shift
        rows, cols, _ = image.shape
        M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
        return cv2.warpAffine(image, M, (cols, rows))

def augment_and_save_images(directory_path, augmenter):
    image_paths = get_image_paths_from_directory(directory_path)
    random.shuffle(image_paths)

    for path in image_paths:
        image = cv2.imread(path)

        # Extracting the filename without extension
        filename = os.path.basename(path)
        base_filename, file_extension = os.path.splitext(filename)

        # Rotations
        cv2.imwrite(os.path.join(directory_path, f"{base_filename}_rotated_-15{file_extension}"), augmenter.rotate_image(image, -15))
        cv2.imwrite(os.path.join(directory_path, f"{base_filename}_rotated_15{file_extension}"), augmenter.rotate_image(image, 15))
        cv2.imwrite(os.path.join(directory_path, f"{base_filename}_rotated_30{file_extension}"), augmenter.rotate_image(image, 30))

        # Translations
        cv2.imwrite(os.path.join(directory_path, f"{base_filename}_translated_right{file_extension}"), augmenter.translate_image(image, 25, 0))
        cv2.imwrite(os.path.join(directory_path, f"{base_filename}_translated_left{file_extension}"), augmenter.translate_image(image, -25, 0))
        cv2.imwrite(os.path.join(directory_path, f"{base_filename}_translated_down{file_extension}"), augmenter.translate_image(image, 0, 25))
        cv2.imwrite(os.path.join(directory_path, f"{base_filename}_translated_up{file_extension}"), augmenter.translate_image(image, 0, -25))
