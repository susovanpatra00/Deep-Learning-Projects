# Traffic Sign Recognition using Convolutional Neural Networks

This project aims to build a convolutional neural network (CNN) model to recognize traffic signs. The dataset used for this project contains images of 43 different types of traffic signs.

## Dataset
The dataset used for this project can be found [here](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign). It consists of images of various traffic signs commonly found on German roads. Each image is labeled with the type of traffic sign it represents.

## Requirements
- Python 3.x
- TensorFlow
- NumPy
- Pandas
- Matplotlib
- PIL (Python Imaging Library)

## Model Architecture
The CNN model consists of several convolutional layers followed by max-pooling layers and dropout layers to prevent overfitting. The final layer uses a softmax activation function to predict the probability distribution over the 43 classes of traffic signs.

## Training
The model is trained using the training dataset, and its performance is evaluated using the validation dataset. The training process involves optimizing the categorical cross-entropy loss function using the Adam optimizer.

## Results
After training for 15 epochs, the model achieved a validation accuracy of approximately 98.71% and a validation loss of approximately 0.05. When tested on unseen data, the model achieved an accuracy score of approximately 95.04%.

## Visualization
The training and validation accuracy and loss curves are plotted using Matplotlib to visualize the model's performance during training.

