"""Step B"""
import numpy as np
class ANN_layer:
    def __init__(self, n_inputs, n_neurons):
        # Matrix shape (inputs, neurons)
        self.weights = np.random.randn(n_inputs, n_neurons)
        
        # Vector shape
        self.bias = np.zeros(n_neurons)
    def forward(self, inputs):
        # Calculate all neurons at once
        x = np.dot(inputs, self.weights) + self.bias
        
        # Vectorized Leaky-ReLU
        y = np.where(x > 0, x, 0.01 * x)
        
        return y

if __name__ == "__main__":
    inputs = np.array([3.0, 5.0, 2.0])

    layer = ANN_layer(n_inputs=3, n_neurons=4)
    output = layer.forward(inputs)

    print(f"Output: {output}")
    print(f"Shape: {output.shape}")