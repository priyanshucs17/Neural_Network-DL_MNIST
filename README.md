# Neural_Network-DL_MNIST
## Problem Statement:
In this assignment, you will build a complete neural network using Numpy. You will implement all the steps required to build a network - __feedforward__, __loss computation__, __backpropagation__, __weight updates__ etc.

You have been provided with some starter code in the notebook below, you have to write the remaining part to create a full-fledged neural network. You only have to write code in cells marked '__#Graded__'  at places where '__# write your code here__'  is mentioned.

You will use the __MNIST__ dataset to train your model to classify handwritten digits between 0-9.

### The assignment is divided into the following sections:
-> Data preparation

-> Feedforward

-> Loss computation

-> Backpropagation

-> Parameter updates

-> Model training and predictions

## Data Preparation:
1. You do not have to write any code in this section (Data Preparation). Please refer to the code provided in the notebook while going through these sections.

2. Firstly, we load the data using the function load_data(). The function data_wrapper() is then applied to the data to get the train and test data in the desired shape. Please note that the code needs to take a batch of data points as the input. Hence, be careful while checking the dimensions.

3. You already know that we have 28x28 greyscale images in the MNIST dataset. Hence, each input image is a vector of length 784. The ground truth labels of a batch are stored in a matrix which is converted to a one-hot matrix. Also, the output of the model is a softmax output of length 10. 

## Feedforward:
There are functions assigned to different subparts of feedforward which we will discuss in some time. But before that, there are some things to take into consideration that will help in implementing the code.

-> The whole data is taken as one batch. No minibatch gradient descent is performed

-> The cumulative input to the layer Z^l is now a step in feedforward

-> The output of the last layer is denoted as H^L instead of P where layer L is the final output layer. Hence, there are L−1 hidden layers.

-> For each layer l, the Z^l is stored as 'activation_memory' and H^l−1, W^l, b^l are stored as 'linear_memory' to use later in backpropagation

### To summarise, the important points to keep in mind are:
The parameters dictionary is getting updated in place at each step.

The memories from L_layer_forward consisting of the tuple memory = (linear_memory, activation_memory) for each layer is used in backpropagation

The backpropagation process will run in a loop from the last layer to the first, and each loop will compute the gradients for Z, H, W, b.

### Important Note on Training:

The training will take about 10-20 minutes with about 2000 iterations, which is a recommended number to achieve good accuracy (> 75%).

PGDML_assignment
