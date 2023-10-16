import os
from BrainTumorClassifier import BrainTumorClassifier  # Assuming BrainTumorClassifier class is in its own .py file.
from dotenv import load_dotenv, find_dotenv

def main():
    load_dotenv(find_dotenv())
    
    # Initialize the classifier
    classifier = BrainTumorClassifier(base_filepath=os.getenv("BASE_FILEPATH"))

    # Setup data generators and model
    train_gen, valid_gen, _ = classifier.setup_data_generators()
    model = classifier.create_model()

    # Train the model
    classifier.train_and_evaluate(model, train_gen, valid_gen, epochs=50)

if __name__ == "__main__":
    main()
