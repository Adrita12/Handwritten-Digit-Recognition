This project focuses on developing a handwritten digit recognition system using machine learning. The goal is to accurately identify digits from images of handwritten characters. The project utilizes the popular MNIST dataset for training and employs a neural network model implemented with TensorFlow.

Workflow
1. Data Loading and Preprocessing:
The MNIST dataset is loaded (optional, as it's included but can be uncommented for training).
Images are normalized to ensure consistent input to the neural network.

2. Neural Network Model:
A simple sequential neural network model is created using TensorFlow.
The model architecture includes a flattening layer, two dense hidden layers with ReLU activation, and an output layer with softmax activation for digit classification.

3. Training:
The model is trained on the MNIST dataset for a specified number of epochs.
The trained model is saved for later use.

4. Model Loading and Inference:
The pre-trained model is loaded from the saved file.
Images of handwritten digits are processed using OpenCV and fed into the model for prediction.

5. Results Display:
Predictions are made for each image, and the results are displayed along with visual representations of the input images using Matplotlib.
