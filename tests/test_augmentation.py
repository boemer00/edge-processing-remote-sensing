import glob
import os
import shutil
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.data.augmentation import ImageAugmenter, augment_and_save_images


def cleanup_augmented_images(directory_path):
    # This function will delete all files that contain '_augmented' in their filename
    for augmented_file in glob.glob(os.path.join(directory_path, "*_augmented*")):
        os.remove(augmented_file)


def setup_test_output_dir(base_dir, output_dir_name="test_output"):
    output_dir_path = os.path.join(base_dir, output_dir_name)
    if os.path.exists(output_dir_path):
        shutil.rmtree(output_dir_path)
    os.makedirs(output_dir_path)
    return output_dir_path


def test_augmentation():
    # Locate the base directory for the test script relative to the 'tests' directory
    test_script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(
        test_script_dir, ".."
    )  # Go up one directory level to the project root
    base_test_dir = os.path.join(base_dir, "tests", "test_data")
    output_dir = setup_test_output_dir(base_dir, "tests/test_output")

    # Clean up any previously augmented images in the output directory
    cleanup_augmented_images(output_dir)

    augmenter = ImageAugmenter(angle=15, x_shift=25, y_shift=25)

    for class_dir in os.listdir(base_test_dir):
        class_path = os.path.join(base_test_dir, class_dir)
        if not os.path.isdir(class_path):  # Skip non-directory files
            continue
        output_class_path = os.path.join(output_dir, class_dir)
        os.makedirs(output_class_path, exist_ok=True)

        augment_and_save_images(class_path, output_class_path, augmenter)

    # Assertions to ensure augmented images are saved correctly
    for class_dir in os.listdir(output_dir):
        if not os.path.isdir(
            os.path.join(output_dir, class_dir)
        ):  # Skip non-directory files
            continue
        augmented_files = os.listdir(os.path.join(output_dir, class_dir))
        assert (
            len(augmented_files) > 0
        ), f"No augmented images found for class {class_dir}"
        for file in augmented_files:
            try:
                assert (
                    "_rotated_" in file or "_translated_" in file
                ), f"File {file} was not augmented"
            except AssertionError as e:
                print(f"Assertion error for file: {file}, {e}")
                raise

    print("All assertions passed. Image augmentation is working as expected.")


if __name__ == "__main__":
    test_augmentation()
