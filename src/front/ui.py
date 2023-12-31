import streamlit as st
from PIL import Image
from io import BytesIO
import base64
import requests
import json
import os
st.set_page_config(layout="wide", page_title="Brain't")

st.write("## Detect tumours in brain MRI scans")
st.write(
    ":brain: Try uploading a brain MRI scan and our model will try and classify it correctly."
)
st.sidebar.write("## Upload")

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
PORT = 8000
MODEL_URL = os.environ.get("MODEL_URL", "http://localhost:8000")

# Download the fixed image
def convert_image(img):
    buf = BytesIO()
    img.save(buf, format="JPEG")
    byte_im = buf.getvalue()
    return byte_im


def fix_image(upload):
    image = Image.open(upload)

    if image.mode not in ["RGB", "L"]:
        image = image.convert("RGB")

    col1.write("Original Image :camera:")
    col1.image(image)

    # Send the image in a POST request to the local server
    # Save the image to a BytesIO buffer
    image_bytes = BytesIO()
    image.save(image_bytes, format='jpeg')

    # Reset the buffer's position to the beginning
    image_bytes.seek(0)

    files = {'file': ('image.jpeg', image_bytes, 'image/jpeg')}
    print(f"{MODEL_URL}/predict")
    response = requests.post(f"{MODEL_URL}/predict", files=files)


    # Display the prediction response
    st.write("Prediction Result:")
    st.json(json.loads(response.text))




col1, col2 = st.columns(2)
my_upload = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if my_upload is not None:
    if my_upload.size > MAX_FILE_SIZE:
        st.error("The uploaded file is too large. Please upload an image smaller than 5MB.")
    else:
        fix_image(upload=my_upload)