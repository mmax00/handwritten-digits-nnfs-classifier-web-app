import pickle
from neuralnetwork.layers import DenseLayer
from neuralnetwork.activation_functions import Tanh
from neuralnetwork.models import Sequential

#Training neural net
with open("./datasets/modified_mnist.pkl", "rb") as f:
    data =  pickle.load(f)

x_train = data["x_train"]
y_train = data["y_train"]
x_test = data["x_test"]
y_test = data["y_test"]

model = Sequential()
model.add(DenseLayer(28*28,40,Tanh()))
model.add(DenseLayer(40,10,Tanh()))

model.fit(x_train,y_train,100,0.1)

model.accuracy(x_test,y_test)
model.predict(x_test[369],y_test[369])

pickle.dump(model,open("./models/trained_model.pkl","wb"))