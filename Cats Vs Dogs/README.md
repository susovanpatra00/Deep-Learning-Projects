# Cats vs Dogs Image Classification

This project demonstrates the implementation of a convolutional neural network (CNN) for the classification of images of cats and dogs. The CNN is built using the TensorFlow library in Python.

## Table of Contents

- [Introduction](#introduction)
- [Neural Network Architecture](#neural-network-architecture)
- [Training](#training)
- [Results](#results)
- [Usage](#usage)
- [Dependencies](#dependencies)


## Introduction

The goal of this project is to build a machine learning model capable of accurately classifying images of cats and dogs. The model is trained on a dataset containing labeled images of cats and dogs, sourced from the Kaggle competition "Dogs vs. Cats". The dataset consists of colored images with varying sizes.

## Neural Network Architecture

The CNN architecture used in this project consists of multiple convolutional and pooling layers followed by fully connected layers. The architecture is designed to learn hierarchical features from the input images, starting from low-level features (e.g., edges, textures) to high-level features. The final layer uses a softmax activation function to output the probability of each image belonging to either the cat or dog class.

## Training

The model is trained using the Adam optimizer with a categorical cross-entropy loss function. During training, data augmentation techniques such as rotation, shearing, and horizontal flipping are applied to enhance the model's generalization ability. Early stopping and learning rate reduction callbacks are used to prevent overfitting and improve convergence.

## Results

After training the model for 10 epochs, it achieves a validation accuracy of approximately 80%. The training and validation loss curves indicate that the model is learning effectively without overfitting.

## Usage

To use this project:
1. Clone the repository or download the provided Python script.
2. Download the dataset from [Kaggle](https://www.kaggle.com/c/dogs-vs-cats) and extract it to a directory named "train" in the project folder.
3. Install the required dependencies using `pip install -r requirements.txt`.
4. Run the Python script to train the neural network.
5. Optionally, modify the hyperparameters or architecture of the model to experiment with different configurations.

## Dependencies

The following dependencies are required to run the project:
- TensorFlow
- NumPy
- Pandas
- Matplotlib
- scikit-learn

You can install the dependencies using pip:

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn
```
