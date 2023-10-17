"""
train module

This module contains the implementation of the function
that trains the model.
"""
import os
import argparse
from dotenv import load_dotenv, find_dotenv
from BrainTumorClassifier import BrainTumorClassifier


def main():
    """
        main function

        This function contains the implementation of the training
         of the model
        """
    load_dotenv(find_dotenv())
    parser = argparse.ArgumentParser(description="Train the model")
    parser.add_argument('--random_state', type=int, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    args = parser.parse_args()
    # Initialize the classifier
    classifier = BrainTumorClassifier(base_filepath=os.getenv("BASE_FILEPATH"),
                                      seed=args.random_state)

    # Setup data generators and model
    train_gen, valid_gen, _ = classifier.setup_data_generators()
    model = classifier.create_model()

    # Train the model
    classifier.train_and_evaluate(model, train_gen, valid_gen, epochs=args.epochs)

if __name__ == "__main__":
    main()
