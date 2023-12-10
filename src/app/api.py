"""Main script: it includes our API initialization and endpoints."""

import io
from http import HTTPStatus
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
import mlflow.tensorflow
from tensorflow.keras.preprocessing import image
import numpy as np
from src import MODEL_RUN_ID
from src.app.schemas import BrainTumorType
from PIL import Image


model = mlflow.tensorflow.load_model(f"runs:/{MODEL_RUN_ID}/model")

# Define application
app = FastAPI(
    title="Brain Tumor Classifier",
    description="This API lets you make predictions on the Brain Tumor Classifier.",
    version="0.1",
)


# Endpoint to list possible classes for the image
@app.get("/list_classes")
async def list_classes():
    """
        lists all the possible classification classes
    """

    class_names = [member.name for member in BrainTumorType]

    return {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"classes": class_names}
    }


# Endpoint to list all experiments
@app.get("/list_experiments")
async def list_experiments():
    """
        lists all the experiments done on the mlflow
    """
    experiments = mlflow.search_runs()

    columns_to_keep = [
        'run_id',
        'experiment_id',
        'status',
        'artifact_uri',
        'start_time',
        'end_time',
        'tags.mlflow.runName'
    ]

    filtered_experiments = experiments.loc[:, columns_to_keep]
    filtered_experiments = filtered_experiments.where(pd.notna(filtered_experiments), None)
    json_data = filtered_experiments.to_dict(orient="records")

    return {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"experiments": json_data}
    }


# Endpoint to predict the class of an image
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
        predicts the class of a given image
    """

    # Ensure the uploaded file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    content = await file.read()
    img = image.img_to_array(Image.open(io.BytesIO(content)).convert("RGB").resize((150, 150)))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction_probabilities = model.predict(img_array)[0]

    predicted_class = BrainTumorType(np.argmax(prediction_probabilities)).name
    result_dict = {tumor.name: float(prob)
                   for tumor, prob in zip(BrainTumorType, prediction_probabilities)}

    return {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
        "data": {"predicted_class": predicted_class, "probabilities": result_dict}
    }
