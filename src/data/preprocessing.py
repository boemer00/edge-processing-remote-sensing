import os
import pandas as pd

def list_files_from_directory(directory_path):
    """Return a list of files from the given directory."""
    return [os.path.join(directory_path, f) for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

def get_data_from_directory(data_path):
    """Generate data list from a directory containing subdirectories of images."""
    data_list = []

    for folder in os.listdir(data_path):
        folder_path = os.path.join(data_path, folder)

        if os.path.isdir(folder_path):
            for image_path in list_files_from_directory(folder_path):
                label = folder
                data_list.append({'image_path': image_path, 'label': label})

    return data_list

def generate_dataframe_from_directory(data_path='../../raw_data/'):
    """Generate a DataFrame from a directory containing subdirectories of images."""
    data_list = get_data_from_directory(data_path)
    return pd.DataFrame(data_list)

def get_image_paths_from_directory(directory_path):
    """Return a list of image paths from the given directory."""
    image_extensions = ['.jpg', '.jpeg', '.png']
    return [os.path.join(directory_path, f) for f in os.listdir(directory_path) if f.endswith(tuple(image_extensions))]

def generate_data_list(data_directory):
    """Generate a list of image paths and their associated labels."""
    data_list = []
    for folder in os.listdir(data_directory):
        folder_path = os.path.join(data_directory, folder)
        if os.path.isdir(folder_path):
            for image_path in get_image_paths_from_directory(folder_path):
                label = folder
                data_list.append({'image_path': image_path, 'label': label})
    return data_list
