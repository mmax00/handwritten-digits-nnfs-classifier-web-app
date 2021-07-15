import numpy as np

class Model:
  def mse(self,y_true,y_pred):
    return np.mean(np.power(y_true-y_pred,2))

  def mse_derivative(self,y_true,y_pred):
    return 2*(y_pred-y_true)/np.size(y_true)

class Sequential(Model):
  def __init__(self):
    self.net = []

  def add(self,layer):
    self.net.append(layer)

  def _forward(self,x):
    output = x
    for layer in self.net:
      output=layer.forward(output)
    return output

  def fit(self,x_train,y_train,epochs,learning_rate):
    print("Training model...")
    for e in range(epochs):
      error = 0
      for x,y in zip(x_train,y_train):
        #feed forward
        output = self._forward(x)

        error += self.mse(y,output)

        #back propatation
        gradient = self.mse_derivative(y,output)
        for layer in reversed(self.net):
          gradient = layer.backward(gradient,learning_rate)
      
      error /= len(x_train)
      print(f"{e+1}/{epochs}, error: {error}")

  def predict(self,x,y=None):
    output=self._forward(x)
    print("Prediction: ", np.argmax(output),end='\t')
    if type(y)!=type(None):
      print("True value: ",np.argmax(y))
    else:
      print("")
    return np.argmax(output)
    
  def accuracy(self,x_test,y_test):
    counter = 0
    for i in range(len(x_test)):
      output = self._forward(x_test[i])
      if np.argmax(output) == np.argmax(y_test[i]):
        counter+=1
    acc = counter/len(x_test)
    print(f"Accuracy: {acc}({acc*100}%)")
