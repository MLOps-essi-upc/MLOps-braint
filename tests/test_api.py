import os
import pytest
from fastapi.testclient import TestClient
from src.app.api import app

@pytest.fixture(scope="module", autouse=True)
def client():
    with TestClient(app) as client:
        return client

def test_possible_classes(client):
    response = client.get("/list_classes")
    assert response.status_code == 200
    data = response.json()
    assert "classes" in data["data"]


def test_list_experiments(client):
    response = client.get("/list_experiments")
    assert response.status_code == 200
    data = response.json()
    assert "experiments" in data["data"]


def test_predict_with_valid_image(client):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, "test_img_1.png")
    files = {"file": (image_path, open(image_path, "rb"), "image/png")}
    response = client.post("/predict", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_class" in data["data"]
    assert "probabilities" in data["data"]


def test_predict_with_invalid_file(client):
    files = {"file": ("text_file.txt", b"some random text", "text/plain")}
    response = client.post("/predict", files=files)
    assert response.status_code == 400
    assert "File must be an image" in response.text
