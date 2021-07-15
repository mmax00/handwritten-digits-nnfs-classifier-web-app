import numpy as np

class Tanh:
  def forward(self,input):
    self.input = input 
    return np.tanh(self.input)

  def backward(self,output_gradient):
    return np.multiply(output_gradient,1-np.tanh(self.input)**2)