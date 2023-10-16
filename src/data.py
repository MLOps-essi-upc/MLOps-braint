import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from dotenv import load_dotenv, find_dotenv
from tensorflow.keras.preprocessing.image import save_img
import json
import cv2
import sys
import argparse

class DataHandler:
    """
    Handles the data loading, augmentation, and saving of the processed data for the brain tumor classification task.
    """

    def __init__(self, base_filepath, seed=42):
        """
        Initializes the DataHandler with the base file path and seed for reproducibility.

        :param base_filepath: str, base file path for the dataset.
        :param seed: int, seed for random number generation.
        """
        self.base_filepath = base_filepath
        self.raw_training_folder = os.path.join(base_filepath, "raw/Training/")
        self.raw_testing_folder = os.path.join(base_filepath, "raw/Testing/")

        # Set seed for reproducibility
        np.random.seed(seed)
        tf.random.set_seed(seed)

        # Define the path for the processed data
        self.processed_data_path = os.path.join(base_filepath, "processed/")
        self.processed_training_folder = os.path.join(self.processed_data_path, "Training/")
        self.processed_testing_folder = os.path.join(self.processed_data_path, "Testing/")
        self.processed_validation_folder = os.path.join(self.processed_data_path, "Validation/")

        # Create directories if they don't exist
        for folder in [self.processed_data_path, self.processed_training_folder, self.processed_testing_folder, self.processed_validation_folder]:
            os.makedirs(folder, exist_ok=True)

    def setup_data_generators(self):
        """
        Sets up the ImageDataGenerators for training, validation, and testing, and saves the augmented images to new directories.

        :return: tuple of (train_generator, valid_generator, test_generator)
        """
        # Define your data augmentation parameters
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2,  # assuming 20% validation split
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True
        )

        valid_test_datagen = ImageDataGenerator(rescale=1./255)

        # Set up generators
        train_generator = train_datagen.flow_from_directory(
            self.raw_training_folder,
            subset='training',
            target_size=(150, 150),
            batch_size=64,
            class_mode='categorical',
        )

        valid_generator = train_datagen.flow_from_directory(
            self.raw_training_folder,
            subset='validation',
            target_size=(150, 150),
            batch_size=64,
            class_mode='categorical',
            shuffle=False,
        )

        test_generator = valid_test_datagen.flow_from_directory(
            self.raw_testing_folder,
            target_size=(150, 150),
            batch_size=64,
            class_mode='categorical',
            shuffle=False,
        )

        return train_generator, valid_generator, test_generator

    def clear_previous_data(self):
        """
        Clears previously processed data to avoid duplication.
        """
        for folder in [self.processed_training_folder, self.processed_testing_folder, self.processed_validation_folder]:
            shutil.rmtree(folder, ignore_errors=True)
            os.makedirs(folder)


def save_generator_output(generator, output_dir, num_samples):
    """
    Saves the output of a data generator to a directory.

    :param generator: The data generator (e.g., for training data).
    :param output_dir: Directory where to save the images.
    :param num_samples: Number of samples to retrieve from the generator.
    """
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

    # Iterate over the generator to retrieve batches
    for i in range(num_samples):
        data_batch, label_batch = next(generator)  # Get a batch from the generator
        for j, (image, label) in enumerate(zip(data_batch, label_batch)):
            filename = f"img_{i * generator.batch_size + j}.png"
            image_path = os.path.join(output_dir, filename)
            save_img(image_path, image)  # Save the image
        
def save_class_indices(class_indices, directory):
    """
    Saves the class indices in a JSON file.

    :param class_indices: Dictionary containing class indices.
    :param directory: Directory where the JSON file should be saved.
    """
    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)

    # Define the path of the JSON file
    json_path = os.path.join(directory, 'class_indices.json')

    # Save dictionary as JSON
    with open(json_path, 'w') as outfile:
        json.dump(class_indices, outfile)

def save_generator_output_with_structure(generator, base_output_dir, total_images_desired):
    """
    Saves the output of a data generator to a directory, preserving the data structure.
    The function stops saving once the desired number of images is reached.

    :param generator: The data generator (e.g., for training data).
    :param base_output_dir: Base directory where to save the images, with class subdirectories.
    :param total_images_desired: Total number of images you want to save.
    """
    class_indices = generator.class_indices
    reverse_class_indices = {v: k for k, v in class_indices.items()}  # Map from class index to class name

    images_saved = 0
    while images_saved < total_images_desired:
        data_batch, label_batch = next(generator)  # Get a batch from the generator
        for image, label_encoded in zip(data_batch, label_batch):
            if images_saved >= total_images_desired:
                return  # Stop saving once we have enough images

            # Process the image (e.g., for noise reduction)
            processed_image = process_image(image)  # Adjust according to your method's input requirements

            class_index = np.argmax(label_encoded)  # Get the class index of the current image
            class_name = reverse_class_indices[class_index]  # Get the class name corresponding to the index

            # Determine the correct output directory
            output_dir = os.path.join(base_output_dir, class_name)
            os.makedirs(output_dir, exist_ok=True)

            # Construct a unique filename for the image and save it
            filename = f"img_{images_saved}_{class_name}.png"  # Unique filename using the saved image count
            image_path = os.path.join(output_dir, filename)
            save_img(image_path, processed_image)  # Save the processed image

            images_saved += 1  # Update the count of images saved

def process_image(image):
    """
    Process the image through noise detection and possible removal.
    
    :param image: Image data to be processed.
    :return: Processed image data.
    """
    if is_noisy(image):
        image = cv2.medianBlur(image, 3)
    return image

def is_noisy(image):
    """
    Determine whether an image is noisy using saturation analysis.

    :param image: Image data to be analyzed.
    :return: Boolean, whether the image is considered noisy.
    """
    # Convert image to HSV color space
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)  # Ensure correct color conversion

    # Calculate histogram of saturation channel
    s = cv2.calcHist([image_hsv], [1], None, [256], [0, 256])

    # Calculate percentage of pixels with saturation >= p
    p = 0.05
    s_perc = np.sum(s[int(p * 255):-1]) / np.prod(image_hsv.shape[0:2])

    # Percentage threshold; above: valid image, below: noise
    s_thr = 0.01
    return s_perc < s_thr  # Note: the condition here was corrected



if __name__ == "__main__":
    # Load the environment variables
    load_dotenv(find_dotenv())

    # Parsing command-line arguments
    parser = argparse.ArgumentParser(description="Prepare data for training")
    parser.add_argument('--total_train_images', type=int, required=True)
    parser.add_argument('--total_valid_images', type=int, required=True)
    parser.add_argument('--total_test_images', type=int, required=True)
    parser.add_argument('--random_state', type=int)
    
    args = parser.parse_args()

    total_train_images = args.total_train_images
    total_valid_images = args.total_valid_images
    total_test_images = args.total_test_images
    seed = args.random_state

    # Initialize the data handler
    handler = DataHandler(base_filepath=os.getenv("BASE_FILEPATH"), seed=seed)

    # Clear any previously processed data
    handler.clear_previous_data()

    # Set up the data generators
    train_gen, valid_gen, test_gen = handler.setup_data_generators()    

    # Save the images, respecting the original data structure
    save_generator_output_with_structure(train_gen, handler.processed_training_folder, total_images_desired=total_train_images)
    save_generator_output_with_structure(valid_gen, handler.processed_validation_folder, total_images_desired=total_valid_images)
    save_generator_output_with_structure(test_gen, handler.processed_testing_folder, total_images_desired=total_test_images)