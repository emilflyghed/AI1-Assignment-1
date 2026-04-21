"""Step A1"""
class NeuronA1:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def forward(self, inputs):
        x = 0.0 # initialize the weighted sum

        # Calculate the weighted sum using a for-loop
        for i in range(len(inputs)):
            x += inputs[i] * self.weights[i]

        x += self.bias # Adding bias

        # Activation function, Leaky-ReLU
        y = x if x > 0 else 0.01 * x

        return y

import numpy as np
"""Step A2"""
class NeuronA2:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def forward(self, inputs):
        # Calculate weighted sum using NumPy vector multiplication
        x = np.dot(inputs, self.weights)

        x += self.bias # Adding bias

        # Activation function, Leaky-ReLU
        y = x if x > 0 else 0.01 * x

        return y


"""Test Zone"""
if __name__ == "__main__":
    weights = [1.0, 2.0, 3.0]
    bias = -1.0
    inputs = [2.0, 3.0, 4.0]

    n1 = NeuronA1(weights, bias)
    result1 = n1.forward(inputs)
    print(f"NeuronA1 output: {result1}")

    n2 = NeuronA2(weights, bias)
    result2 = n2.forward(inputs)
    print(f"NeuronA2 output: {result2}")