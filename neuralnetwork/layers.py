import numpy as np

class DenseLayer:
  def __init__(self,input_size,neurons,activation_function):
    self.weights = np.random.randn(neurons,input_size)
    self.biases = np.random.randn(neurons,1)
    self.activation_function = activation_function
  
  def forward(self,input):
    self.input = input
    output = np.dot(self.weights,self.input)+self.biases
    return self.activation_function.forward(output)

  def backward(self,output_gradient,learning_rate):
    output_gradient = self.activation_function.backward(output_gradient)
    weights_gradient = np.dot(output_gradient,self.input.T)
    self.weights -= learning_rate * weights_gradient
    self.biases -= learning_rate * output_gradient
    return np.dot(self.weights.T,output_gradient)
