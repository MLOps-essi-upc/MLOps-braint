# Model Card
## Model Information
- Model Name: [Insert Model Name]
- Model Version: [Insert Version Number]
- Model Card Created: [Insert Date]

## Model Architecture:

The model is a Sequential Convolutional Neural Network (CNN) designed for the classification of tumours. It consists of the following layers:

1. **Convolutional Layer 1:**
   - Filters: 32
   - Kernel Size: (3, 3)
   - Activation Function: Rectified Linear Unit (ReLU)
   - Input Shape: (150, 150, 3)

2. **MaxPooling Layer 1:**
   - Pool Size: (2, 2)

3. **Convolutional Layer 2:**
   - Filters: 64
   - Kernel Size: (3, 3)
   - Activation Function: ReLU

4. **MaxPooling Layer 2:**
   - Pool Size: (2, 2)

5. **Convolutional Layer 3:**
   - Filters: 128
   - Kernel Size: (3, 3)
   - Activation Function: ReLU

6. **MaxPooling Layer 3:**
   - Pool Size: (2, 2)

7. **Flatten Layer:**
   - Flattens the input to a one-dimensional array.

8. **Dense Layer 1:**
   - Neurons: 512
   - Activation Function: ReLU

9. **Dense Layer 2 (Output Layer):**
   - Neurons: 4 (corresponding to the number of classes)
   - Activation Function: Softmax

## Model Description:

The model is constructed with convolutional layers followed by max-pooling layers, designed to capture hierarchical features in image data. The ReLU activation function is used to introduce non-linearity. The final layers consist of fully connected (dense) layers, with the output layer employing the softmax activation function for multi-class classification: no tumour, glioma, meningioma, petiuitary.

## Input Shape:

The model expects input images of size (150, 150, 3), representing images with a width and height of 150 pixels and three color channels (RGB).

## Training Parameters:

- Loss Function: Categorical Crossentropy
- Optimizer: Adam

## Training Data
An 80-20 percent split was performed on the [training data]([url](https://dagshub.com/norhther/MLOps-braint/src/main/data/raw/Training)) was performed for training and validation of the model. Considering the total size of the training set is 2870, we can expect the training to contain 2296 images and validation 574. The testing set contains 394 images. The data was obtained from [this repository]([url](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri#:~:text=Repo%3A-,GitHub,-Read%20Me%3A)) from Kaggle. The team responsible for curating the data performed some cleaning on the non-tumour class. The cleaning was approved by Dr. Shashikant Ubhe. We normalised all images to 150x150 size. Some data preprocessing techniques were performed on the training data to aid the model generalisability. These techniques include:
- Rotation: Within a range of 15º
- Width Shift: 10% image width
- Height Shift: 10% image height
- Shear Range: 10%
- Zoom Range: 10%
- Horizontal Flipping

## Training Objective
This model's objective is to perform image classification on brain MRI scans. Other uses, such as, image segmentation or using MRI scans of different parts of the body are not considered to be the intended uses.

## Ethical Considerations
- Data Collection and Bias: There are fewer cases of the no tumour class, no characteristic of the real-world proportions.  
- Fairness and Equity: We do not have knowledge of age, gender, or demographic variety within our data. Therefore, we cannot guarantee fairness and equity of the model. 
- Privacy and Data Handling: All images seem to have been anonymised and metadata removed.
- Transparency and Accountability: [Describe the model's transparency initiatives, including documentation, model interpretation, and accountability practices.]
## Intended Use
- Primary Use Case: Brain tumour classification
- Users: This machine learning model is aimed to aid doctors or trained specialists in identifying brain tumours in MRI scans. This model is not intended to be used instead of a qualified practitioner.  

## Model Performance
### Evaluation Metrics:
- Micro-average F1-score: [Insert Metric 1 value]
- Sensitivity: [Insert Metric 2 value]
- Specificity: [Insert Metric 3 value]
- Performance on Test Data: [Provide test data performance results, if available.]

## Limitations
- Known Limitations: [List any known limitations or weaknesses of the model.]
- Out-of-Scope Tasks:
  - Image Segmentation
  - Different types of medical images such as X-rays or CT-scans, or normal pictures of human heads. 
  - MRI scans of different parts of the body  
  
## Ethical Considerations
- Bias and Fairness: [Discuss how bias was addressed during model development and any remaining challenges.]
  
## Responsible AI Practices
- Data Governance: [Describe data governance practices, including data acquisition, labeling, and data usage policies.]
- Model Governance: [Explain how model updates, versioning, and deployment are managed.]
- User Education: [Outline efforts to educate users about the model's capabilities and limitations.]
## References
- [List relevant papers, articles, or documentation related to the model and its development.]

## Contact Information
- Model Developers: [Provide contact information for the team or individuals responsible for the model.]
- Support Email: [Insert an email address for inquiries or support related to the model.]
