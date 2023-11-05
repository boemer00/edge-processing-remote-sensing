import os
import numpy as np
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import load_img, img_to_array

# Constants
IMAGE_SIZE = (224, 224)
RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'raw_data')
CLASSES = ['cloudy', 'desert', 'green_area', 'water']

def load_data():
    images = []
    labels = []

    # Load images and labels
    for index, class_name in enumerate(CLASSES):
        class_dir = os.path.join(RAW_DATA_DIR, class_name)
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            image = load_img(image_path, target_size=IMAGE_SIZE)
            image = img_to_array(image)
            images.append(image)
            labels.append(index)  # Encode labels as integers

    # Convert to numpy arrays and normalize pixel values
    images = np.array(images, dtype='float32') / 255.0
    labels = np.array(labels, dtype='int')

    # Split into training and validation sets
    train_images, val_images, train_labels, val_labels = train_test_split(
        images, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # One-hot encode the labels
    train_labels = to_categorical(train_labels, num_classes=len(CLASSES))
    val_labels = to_categorical(val_labels, num_classes=len(CLASSES))

    return (train_images, train_labels), (val_images, val_labels)

if __name__ == '__main__':
    (train_images, train_labels), (val_images, val_labels) = load_data()
    print(f'Training data shape: {train_images.shape}')
    print(f'Training labels shape: {train_labels.shape}')
    print(f'Validation data shape: {val_images.shape}')
    print(f'Validation labels shape: {val_labels.shape}')
