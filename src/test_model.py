import os
import pytest
import mlflow.tensorflow
from sklearn.metrics import f1_score
from dotenv import load_dotenv, find_dotenv
from BrainTumorClassifier import BrainTumorClassifier

@pytest.fixture
def bt_model():
    load_dotenv(find_dotenv())
    repository = "dagshub.com/norhther/MLOps-braint.mlflow"
    username = os.environ.get('MLFLOW_TRACKING_USERNAME')
    password = os.environ.get('MLFLOW_TRACKING_PASSWORD')
    text = f"https://{username}:{password}@{repository}"
    mlflow.set_tracking_uri(text)
    run_id = "a550a3b1cb024e42b3c7087e24632362"
    return mlflow.tensorflow.load_model(f"runs:/{run_id}/model")


def test_model_accuracy(bt_model):
    load_dotenv(find_dotenv())
    classifier = BrainTumorClassifier(base_filepath=os.getenv("BASE_FILEPATH"))
    _, _, test_generator = classifier.setup_data_generators()
    y_pred_prob = bt_model.predict(test_generator)
    y_pred = y_pred_prob.argmax(axis=1)
    y_true = test_generator.classes

    assert f1_score(y_true, y_pred, average='macro') >= 0.65