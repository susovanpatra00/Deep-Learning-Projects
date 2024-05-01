# MNIST Digit Recognition with Convolutional Neural Networks

This project demonstrates the implementation of a Convolutional Neural Network (CNN) for the MNIST digit recognition task. The CNN model is trained on the MNIST dataset, consisting of handwritten digits from 0 to 9, to classify the input images into their respective classes.

## Overview

The MNIST dataset is a popular benchmark dataset in the field of computer vision and machine learning. It contains 60,000 training images and 10,000 testing images of handwritten digits, each of size 28x28 pixels. The goal of this project is to develop a CNN model capable of accurately recognizing and classifying these handwritten digits.

## Dataset

The MNIST dataset used in this project can be downloaded from [Kaggle](https://www.kaggle.com/code/kanncaa1/convolutional-neural-network-cnn-tutorial/data). It includes both the training and testing sets of handwritten digit images.

## Key Steps

1. **Data Preparation:** The MNIST dataset is loaded and preprocessed. The input images are normalized to have pixel values between 0 and 1, reshaped to a 28x28x1 3D format, and the labels are one-hot encoded.

2. **Model Architecture:** A CNN model is constructed using TensorFlow and Keras. The model consists of convolutional layers with ReLU activation, max-pooling layers, dropout regularization, and fully connected layers. 

3. **Training:** The CNN model is trained on the training dataset using the Adam optimizer and categorical cross-entropy loss function. The training process involves iterating over multiple epochs to optimize the model parameters.

4. **Evaluation:** The trained model is evaluated on the validation dataset to assess its performance. Metrics such as accuracy and loss are monitored during training and validation.

5. **Visualization:** Various visualization techniques are used to analyze the model performance, including plotting training and validation accuracy/loss curves and generating a confusion matrix to visualize classification results.

## Results

The CNN model achieves an accuracy of approximately 98.9% on the validation dataset, demonstrating its effectiveness in accurately classifying handwritten digits. The confusion matrix provides insights into the model's performance across different digit classes.

## Usage

To use this project:

1. Clone the repository or download the provided Python script.
2. Download the MNIST dataset from [Kaggle](https://www.kaggle.com/code/kanncaa1/convolutional-neural-network-cnn-tutorial/data).
3. Ensure that the required dependencies (NumPy, Pandas, Matplotlib, Seaborn, TensorFlow, Keras) are installed.
4. Run the Python script to train and evaluate the CNN model on the MNIST dataset.
5. Optionally, modify the model architecture or hyperparameters to experiment with different configurations.

## Acknowledgements

This project is inspired by and adapted from the work of [suneelbvs](https://github.com/suneelbvs/Deep-Learning-Projects/blob/main/CNN%20Projects/Computer%20Vision%20Tasks/MNIST/Simple%20CNN%20Model%20with%20Nice%20Tutorial.ipynb). Special thanks to them for providing valuable insights and code examples.

