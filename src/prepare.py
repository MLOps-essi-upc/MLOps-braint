from pathlib import Path
from BrainTumorClassifier import BrainTumorClassifier

BASE_FILE_PATH = '../'

classifier = BrainTumorClassifier(base_filepath=BASE_FILE_PATH)

Path(BASE_FILE_PATH + "data/processed").mkdir(exist_ok=True)
train_gen, val_gen, test_gen = classifier.setup_data_generators()

# Do something to run the generators