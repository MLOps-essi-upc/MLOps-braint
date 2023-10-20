# Braint: Brain Tumor Classification

## Overview

"Braint" is an MLOps-driven project developed during our advanced MLOps course. This project adheres to the principles of good practices in machine learning operations. It's designed to classify brain tumors with the following categories: Glioma, Meningioma, No Tumor, and Pituitary.

Our model aims to assist medical professionals by providing a second opinion in identifying and differentiating tumor types, potentially leading to quicker, data-driven decisions in a clinical environment.

### Classes

- **Glioma**
- **Meningioma**
- **No Tumor**
- **Pituitary**

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks for execution on google colab
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment
    │
    ├── src                <- Source code for use in this project.
        ├── BrainTumorClassifier.py    <- Python Class which other scripts use
        │
        ├── data.py        <- Scripts to download or generate data
        │
        ├── train.py       <- Scripts to turn raw data into features for modeling
        │
        ├── evaluate.py    <- Scripts to train models and then use trained models to make predictions
        |
        ├── test_model.py  <- Script to test the ensure accuracy



--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
