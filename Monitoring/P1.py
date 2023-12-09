import evidently
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab, ClassificationPerformanceTab,CatTargetDriftTab
from evidently.options import DataDriftOptions
from evidently.pipeline.column_mapping import ColumnMapping
import os
from PIL import Image
import pandas as pd
import requests


def parse_images_from_folder(folder_path):
    image_data = []

    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(root, filename)

                try:
                    with open(image_path, 'rb') as file:
                        # Prepare the files dictionary with a key ('file' in this case) and the file object
                        files = {'file': (image_path, file, 'image/jpeg')}

                        # Make the POST request with the files parameter
                        response = requests.post(url, files=files)
                        response= response.json()
                        response['data']['predicted_class']=response['data']['predicted_class'].lower()
                        image_data.append(response)

                except Exception as e:
                    print(f"Error reading {filename}: {e}")

    # Convert the list of dictionaries to a Pandas DataFrame
    df = image_data

    return df

def read_images_from_folder(folder_path):
    image_data = []

    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(root, filename)

                try:
                    with Image.open(image_path) as img:
                        image_data.append({'class':root[root.rfind("\\")+1:], 'img':img})

                except Exception as e:
                    print(f"Error reading {filename}: {e}")

    # Convert the list of dictionaries to a Pandas DataFrame
    df = image_data

    return df

if __name__ == "__main__":
    # Specify the path to the folder containing images
    folder_path_new = "C:/Users/anpag/PycharmProjects/pythonProject3/new"  # Change this to the actual path
    folder_path_old = "C:/Users/anpag/PycharmProjects/pythonProject3/Training"
    url = "http://10.4.41.39:8000/predict"


    # Read images from the folder into a DataFrame
    result_old = parse_images_from_folder(folder_path_old)
    result_new = parse_images_from_folder(folder_path_new)
    im_old = read_images_from_folder(folder_path_old)
    im_new = read_images_from_folder(folder_path_new)

    column_mapping = ColumnMapping()
    column_mapping.target = 'class'
    column_mapping.prediction = 'prediction'

    ref_data= pd.DataFrame()
    ref_data['class']=[x['class'] for x in im_old]
    ref_data['prediction'] = [x['data']['predicted_class'] for x in result_old]

    prod_data = pd.DataFrame()
    prod_data['class'] = [x['class'] for x in im_new]
    prod_data['prediction'] = [x['data']['predicted_class'] for x in result_new]

    classification_performance_report = Dashboard(tabs=[ClassificationPerformanceTab()])
    classification_performance_report.calculate(ref_data,prod_data,column_mapping=column_mapping)
    classification_performance_report.save("model_performance.html")

    target_drift_dashboard = Dashboard(tabs=[CatTargetDriftTab(verbose_level=1)])
    target_drift_dashboard.calculate(ref_data, prod_data,column_mapping=column_mapping)
    target_drift_dashboard.save("target_drift.html")
