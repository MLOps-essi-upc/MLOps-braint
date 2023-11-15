import os
import mlflow.tensorflow
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


MODEL_RUN_ID = "a550a3b1cb024e42b3c7087e24632362"

repository = "dagshub.com/norhther/MLOps-braint.mlflow"
username = os.environ.get('MLFLOW_TRACKING_USERNAME')
password = os.environ.get('MLFLOW_TRACKING_PASSWORD')
text = f"https://{username}:{password}@{repository}"
mlflow.set_tracking_uri(text)
