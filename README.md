This program uses stochastic gradient descent to train BP neural network
It uses a C++ program to train a neural network and uses MATLAB to display results.


/*** How to run this program ***/
1. Choose appropriate parameters in '../data/INPUTPARAMETERS.txt'
2. Open MATLAB and main.m

/*** Instructions for tuning of parameters ***/
training_data: Relative path for training data

training_labels: Relative path for training labels

test_data: Relative path for test data

test_labels: Relative path for test labels

input_count: Number of input nodes, should be equal to number of pixels of training data (MNIST: 784)

output_count: Number of output nodes, should be equal to number of classes of training data (MNIST: 10)

layer_size: Each number denotes the number of nodes in each layer, from the 1st layer to the last layer

bias: bias

momentum: momentum

learning_rate: learning rate

max_epoch_count: Maximum number of epochs/iterations

error_threshold: Error threshold for training sets

batch_size: Number of images in a single batch for stochastic gradient descent training

activation_type: Activation function (sigmoid, hyperbolic or relu)

weight_changes: Each row denotes a simple tuple (layer#, neuron#), track weight changes of a node

activation_changes: Each row denotes a simple tuple (iter#, layer#, neuron#), track activation changes of a node
