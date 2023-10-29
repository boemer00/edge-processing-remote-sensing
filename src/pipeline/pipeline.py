from data.preprocessing import generate_data_list
from data.augmentation import ImageAugmenter, augment_and_save_images
import os

# def main_pipeline(data_path='../../raw_data/', target_folder='desert'):
#     target_path = os.path.join(data_path, target_folder)

#     # Use the ImageAugmenter
#     augmenter = ImageAugmenter()

#     # Augment and save images
#     augment_and_save_images(target_path, augmenter)

#     # Generate the data list
#     data_list = generate_data_list(data_path)
#     return data_list

# if __name__ == "__main__":
#     main_pipeline()
