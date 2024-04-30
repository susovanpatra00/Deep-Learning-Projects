# Image Classification on Cifar10

This project demonstrates how to build and train a convolutional neural network (CNN) for image classification using the Cifar10 dataset and the Keras library.

## Dataset

The Cifar10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images. The classes are mutually exclusive and include:

1. airplane
2. automobile
3. bird
4. cat
5. deer
6. dog
7. frog
8. horse
9. ship
10. truck

## Project Structure

The project consists of the following files:

- `Classification.ipynb`: Jupyter Notebook containing the code for data preprocessing, model creation, training, and evaluation.

## Dependencies

The project requires the following dependencies:

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- TensorFlow
- Keras

You can install the required packages using pip:

```bash
pip install numpy pandas matplotlib tensorflow keras
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/your-username/image-classification-cifar10.git
```

2. Navigate to the project directory:

```bash
cd image-classification-cifar10
```

3. Open the Jupyter Notebook:

```bash
jupyter notebook Classification.ipynb
```

4. Run the notebook cells sequentially to preprocess the data, create the model, train it, and evaluate its performance.

## Model Architecture

The CNN model architecture used in this project consists of the following layers:

- Convolutional layers with ReLU activation and batch normalization
- Max pooling layers
- Dropout layers for regularization
- Flatten layer to convert 2D feature maps to 1D feature vectors
- Dense layers with ReLU activation
- Output layer with softmax activation for multi-class classification

The model is compiled with the categorical cross-entropy loss function and the Adam optimizer. Accuracy is used as the evaluation metric.

## Results

After training the model for 50 epochs with data augmentation and early stopping, the model achieves an accuracy of 87.39% on the test set. This demonstrates the effectiveness of the CNN architecture in classifying images from the Cifar10 dataset.

## Acknowledgments

The Cifar10 dataset is widely used for benchmarking image classification models and is commonly used in educational settings. The dataset was collected by Alex Krizhevsky, Vinod Nair, and Geoffrey Hinton.
