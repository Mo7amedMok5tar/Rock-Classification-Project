# Rock Classification Project

This project builds a deep learning model to classify different types of rocks based on images. The model is trained using a dataset of rock images and the app leverages this model to predict the type of rock for a given image.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Data Preparation](#data-preparation)
   - [Loading the Dataset](#loading-the-dataset)
   - [Data Augmentation](#data-augmentation)
3. [Data Visualization](#data-visualization)
   - [Image Distribution](#image-distribution)
   - [Pairplots to Understand Feature Relationships](#pairplots-to-understand-feature-relationships)
4. [Data Splitting](#data-splitting)
   - [Splitting Data into Training, Validation, and Test Sets](#splitting-data-into-training-validation-and-test-sets)
5. [Model Building](#model-building)
   - [Constructing the Deep Learning Model](#constructing-the-deep-learning-model)
   - [Model Architecture](#model-architecture)
6. [Model Training](#model-training)
   - [Compiling the Model](#compiling-the-model)
   - [Training the Model with Callbacks](#training-the-model-with-callbacks)
7. [Model Evaluation](#model-evaluation)
   - [Evaluating Performance using Confusion Matrix](#evaluating-performance-using-confusion-matrix)
   - [Accuracy and Loss Plots](#accuracy-and-loss-plots)
8. [User Interface](#user-interface)
   - [Streamlit-based Web App to Classify Rocks](#streamlit-based-web-app-to-classify-rocks)

## Project Overview

This project involves building and deploying a machine learning model to classify rock types. The model is trained using a dataset of images of different rock types, specifically **Baryte**, **Calcite**, **Fluorite**, and **Pyrite**. The goal is to predict the rock type based on the input image.

## Data Preparation

### 1. Loading the Dataset
The dataset is loaded from a collection of rock images categorized into different folders based on rock types. The dataset is preprocessed for model training.

### 2. Data Augmentation
- For the training set, we apply data augmentation using `ImageDataGenerator` from `Keras`. This includes transformations such as rotation, width/height shift, zooming, and flipping to artificially expand the training dataset and improve model generalization.
- For the validation and test sets, we only apply rescaling, to normalize the images.

## Data Visualization

### 1. Image Distribution
- The distribution of images across different classes (rock types) is visualized using bar plots.

### 2. Pairplots to Understand Feature Relationships
- Pairplots are used to explore relationships between pixel values (features) and visualize patterns across the different rock types.

## Data Splitting

The dataset is split into training, validation, and test sets. The data is divided into 80% for training, 10% for validation, and 10% for testing, ensuring that each class is equally represented across the splits.

## Model Building

### 1. Constructing the Deep Learning Model
The model is constructed using **Keras** and **TensorFlow**. The architecture includes **Convolutional Neural Networks (CNNs)**, which are particularly effective for image classification tasks.

### 2. Model Architecture
The model is built using a pre-trained **InceptionV3** architecture for transfer learning. We utilize `InceptionV3` without the top layers, keeping only the convolutional base.
Custom layers are added on top of the base model, including:
- **Global Average Pooling** to reduce the dimensionality.
- **Dense layers** with **ReLU activations**, **Batch Normalization**, and **Dropout** to prevent overfitting.
- The final output layer uses the **softmax activation function** for multi-class classification.

## Model Training

### 1. Compiling the Model
The model is compiled using the **Adam optimizer** and **categorical cross-entropy loss**, appropriate for multi-class classification tasks. The evaluation metric used is accuracy.

### 2. Training the Model with Callbacks
The model is trained using the `fit` method with the training and validation data generators. Early stopping and learning rate reduction callbacks are used to monitor the validation loss and prevent overfitting.

## Model Evaluation

### 1. Accuracy and Loss Plots
Accuracy and loss curves for both training and validation sets are plotted to visualize the model's learning progress.

### 2. Confusion Matrix
A confusion matrix is generated for both the validation and test sets to evaluate the model’s classification performance, showing how well the model is predicting each rock type.

## User Interface

### Streamlit-based Web App to Classify Rocks

The web app allows users to upload an image of a rock, and the model will classify the type of rock.

#### Inputs:
- Users upload an image of a rock through the web interface.

#### Predictions:
- The app predicts the rock type (Baryte, Calcite, Fluorite, or Pyrite) based on the uploaded image and displays the prediction along with the model's confidence score.

## Conclusion

This project demonstrates how deep learning can be used for rock classification based on image data. The model was trained using a collection of labeled rock images and is evaluated with various performance metrics. Additionally, a user-friendly Streamlit app was built to allow users to upload images and get predictions in real-time.

**Note**: To run the app, ensure that the trained model (`rock_classifier.h5`) and necessary image processing files are in the correct path.
