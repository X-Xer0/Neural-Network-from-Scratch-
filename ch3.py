import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons): # Layer initialization
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons) # Initialize weights and biases
        self.biases = np.zeros((1, n_neurons))

def forward(self, inputs):
    self.output = np.dot(inputs, self.weights) + self.biases

X, y = spiral_data(samples=100, classes=3) # Create dataset

dense1 = Layer_Dense(2, 3) # Create Dense layer with 2 input features and 3 output values
dense1.forward(X) # Perform a forward pass of our training data through this layer

print(dense1.output[:5])