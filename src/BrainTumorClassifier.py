"""
BrainTumorClassifier module

This module contains the implementation of the BrainTumorClassifier class for
detecting brain tumors in medical images.
"""
import os
import numpy as np
import tensorflow as tf
import mlflow
import mlflow.tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
from codecarbon import EmissionsTracker


class BrainTumorClassifier:
    """
        BrainTumorClassifier class

        This class provides methods and attributes for detecting brain tumors in
        medical images.
        """

    def __init__(self, base_filepath, seed=42):
        self.FILEPATH = base_filepath
        self.TRAINING_FOLDER = os.path.join(base_filepath, "data/processed/Training/")
        self.TESTING_FOLDER = os.path.join(base_filepath, "data/processed/Testing/")
        self.MODEL_FILEPATH = os.path.join(base_filepath, "models/")
        self.CLASSES = ["glioma_tumor", "menigioma_tumor", "no_tumor", "pituitary_tumor"]

        # Set seed for reproducibility
        np.random.seed(seed)
        tf.random.set_seed(seed)

    # def mount_drive(self):
    #     """
    #     Mounts the Google Drive.
    #     """
    #     from google.colab import drive
    #     drive.mount('/content/drive', force_remount=True)

    def setup_data_generators(self):
        """
        Sets up the ImageDataGenerators for training, validation, and testing.

        Returns:
            tuple: train_generator, valid_generator, test_generator
        """
        # valid_datagen = ImageDataGenerator(rescale=1. / 255)
        test_datagen = ImageDataGenerator(rescale=1. / 255)

        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            validation_split=0.2,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True
        )

        train_generator = train_datagen.flow_from_directory(
            self.TRAINING_FOLDER,
            subset='training',
            target_size=(150, 150),
            batch_size=64,
            class_mode='categorical'
        )

        valid_generator = train_datagen.flow_from_directory(
            self.TRAINING_FOLDER,
            subset='validation',
            target_size=(150, 150),
            batch_size=64,
            class_mode='categorical',
            shuffle=False
        )

        test_generator = test_datagen.flow_from_directory(
            self.TESTING_FOLDER,
            target_size=(150, 150),
            batch_size=64,
            class_mode='categorical',
            shuffle=False
        )

        return train_generator, valid_generator, test_generator

    def create_model(self):
        """
        Defines the CNN architecture.

        Returns:
            Sequential: A compiled TensorFlow model.
        """
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(512, activation='relu'),
            Dense(4, activation='softmax')
        ])

        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def setup_mlflow(self, experiment_name="braint",
                     repository="dagshub.com/norhther/MLOps-braint.mlflow"):
        """
        Sets up MLflow for experiment tracking.
        Args:
            experiment_name: Name of the MLflow experiment
        """
        # mlflow.set_tracking_uri('file://' + self.FILEPATH + "/mlflow_experiments/")
        username = os.environ.get('MLFLOW_TRACKING_USERNAME')
        password = os.environ.get('MLFLOW_TRACKING_PASSWORD')
        mlflow.set_tracking_uri(f"https://{username}:{password}@{repository}")

        if mlflow.get_experiment_by_name(experiment_name) is None:
            mlflow.create_experiment(experiment_name)
        mlflow.tensorflow.autolog()

    def log_used_files(self, phase, generator):
        """
        Logs the files used during different phases (training, validation, testing) in MLflow.

        Args:
            phase (str): The phase during which the files were used ('training',
            'validation', or 'testing').
            generator (DirectoryIterator): Generator containing the filenames.
        """
        filenames = generator.filenames
        artifact_path = f"{phase}_files.txt"
        with open(artifact_path, 'w', encoding='utf-8') as f:
            for item in filenames:
                f.write(f"{item}\n")

        mlflow.log_artifact(artifact_path)

        # delete the file after logging it
        os.remove(artifact_path)

    def train_and_evaluate(self, model, train_generator, valid_generator, epochs=10,
                           experiment_name="braint"):
        """
        Trains the provided model and evaluates its performance. Metrics
         and artifacts are logged with MLflow.

        Args:
            model (Sequential): The TensorFlow model to train.
            train_generator (DirectoryIterator): Generator for the training data.
            valid_generator (DirectoryIterator): Generator for the validation data.
            epochs (int): The number of epochs to train the model. Default is 10.
            experiment_name (str): The name of the MLflow experiment. Default is "braint".
        """
        self.setup_mlflow(experiment_name)
        tracker = EmissionsTracker()
        tracker.start()
        with mlflow.start_run(
                experiment_id=mlflow.get_experiment_by_name(experiment_name).experiment_id):
            # Log the files used for training, validation, and testing
            self.log_used_files("training", train_generator)
            self.log_used_files("validation", valid_generator)

            tracker.start_task("Train the model")
            history = model.fit(train_generator, epochs=epochs, validation_data=valid_generator)
            tracker.stop_task()

            # Setup for test data evaluation
            # Get the test generator
            _, _, test_generator = self.setup_data_generators()
            self.log_used_files("testing", test_generator)

            # After training finishes, evaluate on the test set
            tracker.start_task("Evaluate the model")
            y_pred_prob = model.predict(test_generator)
            y_pred = y_pred_prob.argmax(axis=1)
            tracker.stop_task()
            y_true = test_generator.classes

            report = classification_report(y_true, y_pred, target_names=self.CLASSES,
                                           output_dict=True)
            accuracy = accuracy_score(y_true, y_pred)

            # Log metrics and artifacts
            mlflow.log_metrics({"accuracy": accuracy})

            report_str = '\n'.join([f'{key}: {item}' for key, item in report.items()])
            mlflow.log_text(report_str, "classification_report.txt")

            # Plot ROC Curve and log as an artifact
            # Considering 4 classes as per `self.CLASSES`
            binarized_y_true = label_binarize(y_true,
                                              classes=[0, 1, 2, 3])
            self.plot_and_log_roc_curve(binarized_y_true, y_pred_prob)

            # Save the model as an artifact
            mlflow.tensorflow.log_model(model, "models")

            # specify your desired save path here
            model_save_path = os.path.join(self.MODEL_FILEPATH, "model")
            model.save(model_save_path)

            # Ensure the MLflow run is ended
            mlflow.end_run()
            tracker.stop()
        return history, report

    def evaluate_on_test(self, model, test_generator, experiment_name="braint"):
        """
        Evaluate the model on test data and log metrics to MLflow.

        Args:
            model (tf.keras.Model): The TensorFlow model to evaluate.
            test_generator (DirectoryIterator): Generator for the test data.
            experiment_name (str): The name of the MLflow experiment. Default is "braint".
        """
        # Ensure MLflow is set up
        self.setup_mlflow(experiment_name)

        with mlflow.start_run(
                experiment_id=mlflow.get_experiment_by_name(experiment_name).experiment_id):
            # Log the files used for testing
            self.log_used_files("testing", test_generator)

            # Evaluate the model on the test set
            scores = model.evaluate(test_generator)
            metrics = {'test_' + metric: value for metric, value \
                       in zip(model.metrics_names, scores)}

            # Log metrics
            mlflow.log_metrics(metrics)

            # Predict and generate classification report and confusion matrix, if needed
            y_pred_prob = model.predict(test_generator)
            y_pred = y_pred_prob.argmax(axis=1)
            y_true = test_generator.classes

            report = classification_report(y_true, y_pred,
                                           target_names=self.CLASSES, output_dict=True)
            report_str = '\n'.join([f'{key}: {item}' for key, item in report.items()])
            mlflow.log_text(report_str, "classification_report_test.txt")

            # Optionally, log ROC curve for test data as well, similar to training
            # Considering 4 classes as per `self.CLASSES`
            binarized_y_true = label_binarize(y_true,
                                              classes=[0, 1, 2, 3])
            self.plot_and_log_roc_curve(binarized_y_true, y_pred_prob)

            # Ensure the MLflow run is ended
            mlflow.end_run()

    def plot_and_log_roc_curve(self, y_true, y_pred_prob, save=False):
        """
        Plots the ROC curve for the multi-class scenario and logs the image as an MLflow artifact.

        Args:
            y_true (np.array): True binary labels in binary indicator format.
            y_pred_prob (np.array): Predicted probabilities for each class.
        """
        # Compute ROC curve and ROC area for each class
        fpr = {}
        tpr = {}
        roc_auc = {}
        for i in range(len(self.CLASSES)):
            fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot all ROC curves
        plt.figure()
        for i, color in zip(range(len(self.CLASSES)), ['blue', 'red', 'green', 'black']):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label=f'ROC curve of class {self.CLASSES[i]} (area = {roc_auc[i]:0.2f})')

        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Some extension of Receiver operating characteristic to multi-class')
        plt.legend(loc="lower right")
        plt.savefig("roc_curves.png")

        mlflow.log_artifact("roc_curves.png")
        if not save:
            os.remove("roc_curves.png")
