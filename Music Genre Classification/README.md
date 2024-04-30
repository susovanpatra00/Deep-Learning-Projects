# Music Genre Classification using Neural Networks

This project aims to classify music genres using machine learning techniques, specifically neural networks. It utilizes a dataset containing audio features extracted from music clips, such as tempo, spectral centroid, and others. These features are then used to train a neural network model to predict the genre of a given music clip.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Feature Engineering](#feature-engineering)
- [Model Training](#model-training)
- [Results](#results)
- [Usage](#usage)
- [Dependencies](#dependencies)

## Introduction

Music genre classification is a fundamental task in the field of music information retrieval (MIR). It involves the automatic categorization of music clips into predefined genres based on their audio features. This project explores the use of neural networks for this purpose.

## Dataset

The dataset used in this project consists of audio features extracted from music clips. It includes attributes such as tempo, spectral centroid, chroma features, and others. The dataset is divided into two parts: a training set and a test set.

- Training Dataset: `features_3_sec.csv`
- Test Dataset: `features_30_sec.csv`

## Feature Engineering

Before training the model, some preprocessing steps are performed on the dataset:

1. Removal of unnecessary columns ('filename' and 'length').
2. Standardization of feature values using `StandardScaler`.
3. Encoding of target labels into integers using `LabelEncoder`.

## Model Training

A neural network model is built using TensorFlow and Keras. The architecture of the model consists of several dense layers with ReLU activation functions, interspersed with dropout layers to prevent overfitting. The model is trained using the training dataset and optimized using the Adam optimizer.

Early stopping is implemented to prevent overfitting and improve generalization performance. The model's training progress is monitored using the validation dataset.

## Results

After training the model, it is evaluated on the test dataset to assess its performance. The accuracy achieved on the test dataset is approximately 98.7%, indicating the model's ability to accurately classify music genres. Additionally, the model achieved a validation accuracy of 90.76%.

## Usage

To use this project:

1. Clone the repository.
2. Install the necessary dependencies listed in the `requirements.txt` file.
3. Run the provided Jupyter Notebook or Python script to train and evaluate the model.
4. Optionally, modify the model architecture or hyperparameters to experiment with different configurations.

## Dependencies

The following dependencies are required to run the project:

- Python 3.x
- TensorFlow
- Keras
- NumPy
- pandas
- matplotlib
- librosa

You can install the dependencies using pip:  ```pip install -r requirements.txt```
