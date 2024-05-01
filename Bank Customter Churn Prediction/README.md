# Bank Customer Churn Prediction

This project aims to predict whether a bank customer will churn (leave the bank) using artificial neural networks (ANNs). The dataset used for this project is the "Churn_Modelling.csv" file.

## Introduction

The objective of this project is to build a binary classification model to predict whether a bank customer is likely to churn based on various features such as their geographic location, gender, credit score, balance, etc. The model is built using ANNs implemented with the TensorFlow library.

## Data Preprocessing

The dataset contains both numerical and categorical features. Categorical features like "Geography" and "Gender" are one-hot encoded to convert them into a suitable format for training the model. Standard scaling is applied to normalize the numerical features.

## Model Architecture

The ANN model consists of multiple dense layers with ReLU activation functions. Dropout regularization is applied to prevent overfitting. The output layer uses a sigmoid activation function since it's a binary classification problem.

## Training

The model is trained using the Adam optimizer and binary cross-entropy loss function. Early stopping is implemented to halt training when the validation loss stops decreasing. The training history is monitored to analyze the model's performance.

## Results

After training the model for 1000 epochs, it achieves an accuracy of approximately 85% on the test data. The model's performance is visualized using accuracy plots, and the confusion matrix provides insights into the model's predictive capabilities.

## Conclusion

This project serves as a practice exercise to understand the fundamentals of deep learning and neural networks. Despite being a simple project, it demonstrates the effectiveness of ANNs in solving real-world classification problems.

Feel free to experiment with different architectures, hyperparameters, and preprocessing techniques to further improve the model's performance.

