# This file has code to train deep learning model from scratch. A lot of this code is motivated from coursera's deep learning specialization course.
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt

def initialize_parameters(layers_dims: list[int], init_type=None) -> Dict[str, np.ndarray]:
    """
    Initializes the weights and biases for a deep neural network with a given architecture.
    Parameters:
    layers_dims : list[int]
        List containing the dimensions of each layer in the network, including the input and output layers.
        Example: [input_size, hidden_layer_1_size, ..., output_size].

    Returns:
    --------
    parameters : Dict[str, np.ndarray]
        Dictionary containing initialized weight matrices and bias vectors for each layer:
        - "W1", "W2", ..., "WL" represent the weight matrices for layers 1 through L.
        - "b1", "b2", ..., "bL" represent the bias vectors for layers 1 through L.
        The weight matrix Wl has shape (layers_dims[l], layers_dims[l-1]), and the bias vector bl has shape (layers_dims[l], 1).

    Weights are initialized using a small random number (multiplied by 0.01) from a standard normal distribution.
    Biases are initialized randomly but not scaled.
    """
    L: int = len(layers_dims)  # number of layers in the network (including zeroth layer)
    parameters = dict()

    # initializes weights and biases based on number of layers
    for l in range(1, L):

        if init_type == 'he_uniform':
            # Get He Uniform initialization
            limit = np.sqrt(6.0 / layers_dims[l - 1])
            parameters[f"W{l}"] = np.random.uniform(low=-limit, high=+limit, size=(layers_dims[l], layers_dims[l - 1]))
        else:
            parameters[f"W{l}"] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(2.0 / layers_dims[l-1]) #0.01
        parameters[f"b{l}"] = np.zeros((layers_dims[l], 1))

    return parameters


def softmax_activation(Z: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Implements the softmax activation function.
    Parameters:
    Z : np.ndarray
        The input to the activation function, i.e. the linear part of the forward propagation (W * A_prev + b).
    Returns:
    A : np.ndarray
        The output of the softmax activation function, computed as 1 / (1 + exp(-Z)).
    cache : Tuple[np.ndarray, np.ndarray]
        A tuple containing the output `A` and input `Z` for efficient backward propagation.
    """
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    A = exp_Z / np.sum(exp_Z, axis=0, keepdims=True)  # Softmax calculation
    cache = Z
    return A, cache


def relu_activation(Z: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Implements the ReLU (Rectified Linear Unit) activation function.
    Parameters:
    Z : np.ndarray
        The input to the activation function, i.e. the linear part of the forward propagation (W * A_prev + b).
    Returns:
    A : np.ndarray
        The output of the ReLU activation function, computed as max(0, Z).
    cache : Tuple[np.ndarray, np.ndarray]
        A tuple containing the output `A` and input `Z` for efficient backward propagation.
    """
    A = np.maximum(Z, 0)
    cache = Z
    return A, cache


def linear_forward(A: np.ndarray, W: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Implements the linear part of forward propagation for a layer (i.e., Z = W * A_prev + b).
    Parameters:
    A : np.ndarray
        Activations from the previous layer (or input data for the first layer).
    W : np.ndarray
        Weights matrix of the current layer.
    b : np.ndarray
        Bias vector of the current layer.

    Returns:
    Z : np.ndarray
        The input to the activation function (the linear part of the forward propagation).
    cache : Tuple[np.ndarray, np.ndarray, np.ndarray]
        A tuple containing `A`, `W`, and `b` for efficient backward propagation.
    """
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    return Z, cache


def linear_activation_forward(A_prev: np.ndarray, W: np.ndarray, b: np.ndarray, activation: str) -> Tuple[np.ndarray, Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]]:
    """
   Implements forward propagation for the linear part and the activation (softmax or ReLU) for a layer.
   Parameters:
   A_prev : np.ndarray
       Activations from the previous layer (or input data for the first layer).
   W : np.ndarray
       Weights matrix of the current layer.
   b : np.ndarray
       Bias vector of the current layer.
   activation : str
       The activation function to use ("softmax" or "relu").
   Returns:
   A : np.ndarray
       The output of the activation function for the current layer.
   cache : Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]
       A tuple containing both the linear cache (A_prev, W, b) and the activation cache (A, Z) for efficient backward propagation.
   """
    Z, linear_cache = linear_forward(A_prev, W, b)  # This "linear_cache" contains (A_prev, W, b)

    if activation == 'softmax':
        A, activation_cache = softmax_activation(Z)
    elif activation == 'relu':
        A, activation_cache = relu_activation(Z)
    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward(X: np.ndarray, parameters: Dict[str, np.ndarray], layers_dims: List[int]) -> Tuple[np.ndarray, List[Tuple]]:
    """
    Implements forward propagation for the entire deep neural network, from input X to the output layer.
    Parameters:
    X : np.ndarray
        Input data of shape (input_size, number_of_examples).

    parameters : Dict[str, np.ndarray]
        Dictionary containing the initialized weights and biases for each layer.
        Keys are in the form "W1", "b1", ..., "WL-1", "bL-1".

    layers_dims : List[int]
        List containing the dimensions of each layer, including input, hidden, and output layers.
        Example: [input_size, hidden_layer_1_size, ..., output_layer_size].

    Returns:
    A_out_layer : np.ndarray
        The output of the final (output) layer, which is used for predictions.

    caches_list : List[Tuple]
        A list of caches, where each cache contains the linear and activation caches for each layer,
        used for backward propagation.
    """
    A = X
    # number of layers with zeroth layer
    L = len(parameters) // 2
    caches_list = []

    # Calculate activation functions for L-1 layers (exclude the last layer as that's the output layer)
    for l in range(1, L):
        A_prev = A
        W = parameters[f"W{l}"]
        b = parameters[f"b{l}"]
        A, cache = linear_activation_forward(A_prev, W, b, "relu")
        caches_list.append(cache)

    # Calculate Activation for the output layer (L-1 th layer)
    W = parameters[f"W{L}"]
    b = parameters[f"b{L}"]
    A_out_layer, cache = linear_activation_forward(A, W, b, "softmax")
    caches_list.append(cache)

    return A_out_layer, caches_list


def compute_cost(A_out_layer, Y):
    # we use the multiclass cross-entropy loss formula where softmax is the activation function:
    # formula is -1/m * sum(yk(i)*log(y_predk(i)))

    # To avoid log(0), we can clip the predicted probabilities
    # A_out_layer = np.clip(A_out_layer, 1e-15, 1 - 1e-15)

    m = Y.shape[0]
    cost = (-1 / m) * np.sum(Y * np.log(A_out_layer), axis=1, keepdims=True)
    total_cost = np.sum(cost)
    return total_cost, cost


def linear_backward(dZ, cache):
    # Calculate backward prop for linear functions
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1 / m) * np.dot(dZ, A_prev.T)
    db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def relu_backward(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ


def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)

    elif activation == "softmax":
        dZ = dA  # For softmax, the gradient of softmax activation is dA itself

    dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def L_model_backward(A_out_layer, Y, caches):
    grads = dict()
    L = len(caches)  # the number of layers
    m = A_out_layer.shape[1]
    Y = Y.reshape(A_out_layer.shape)

    # Initializing the backpropagation for multi-class classification
    dA_out_layer = A_out_layer - Y

    current_cache = caches[L - 1]  # Last Layer
    grads["dA" + str(L - 1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dA_out_layer,
                                                                                                      current_cache,
                                                                                                      "softmax")

    # Loop from l=L-2 to l=0
    for l in reversed(range(L - 1)):
        # lth layer: (RELU -> LINEAR) gradients.
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache,
                                                                    activation="relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent

    Arguments:
    parameters -- python dictionary containing your parameters
    grads -- python dictionary containing your gradients, output of L_model_backward

    Returns:
    parameters -- python dictionary containing your updated parameters
                  parameters["W" + str(l)] = ...
                  parameters["b" + str(l)] = ...
    """
    L = len(parameters) // 2  # number of layers in the neural network

    # Update rule for each parameter
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
    return parameters


def L_layer_model(X, Y, layers_dims, parameters, learning_rate, num_iterations, print_cost=False):
    costs = []  # keep track of cost
    total_costs = []

    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation:
        A_out_layer, caches = L_model_forward(X, parameters, layers_dims)

        # Compute cost.
        total_cost, cost = compute_cost(A_out_layer, Y)

        # Backward propagation.
        grads = L_model_backward(A_out_layer, Y, caches)

        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the cost every 100 training example
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, total_cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
            total_costs.append(total_cost)

    # plot the cost
    plt.plot(total_costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters