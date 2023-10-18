"""
evaluation module

This module contains the implementation of the function
that evaluates the model.
"""
import os
import tensorflow as tf
from dotenv import load_dotenv, find_dotenv
from BrainTumorClassifier import BrainTumorClassifier


def main():
    """
    main function

    This function contains the implementation of the evaluation
     of the model
    """
    load_dotenv(find_dotenv())

    print(os.getenv("BASE_FILEPATH"))

    # Initialize the classifier
    classifier = BrainTumorClassifier(base_filepath=os.getenv("BASE_FILEPATH"))

    # Load the trained model
    model = tf.keras.models.load_model(os.path.join(os.getenv("MODEL_FILEPATH"),
                                                    'model'))

    # Setup for test data evaluation
    _, _, test_generator = classifier.setup_data_generators()  # Get the test generator

    # Evaluate the model on test data
    # The evaluate_on_test function should handle evaluation and logging metrics to MLflow
    classifier.evaluate_on_test(model, test_generator)


if __name__ == "__main__":
    main()
