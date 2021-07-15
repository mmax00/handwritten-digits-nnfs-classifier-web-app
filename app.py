from neuralnetwork.layers import DenseLayer
from neuralnetwork.activation_functions import Tanh
from neuralnetwork.models import Sequential

from flask import Flask, render_template, request
import json
from urllib.parse import unquote_plus
import pickle

app = Flask(__name__)

model =  pickle.load(open("./models/trained_model.pkl","rb"))

@app.route('/')
def index():
    return render_template('index.html')


def transformData(data):
    jsonObject = json.loads(data)
    drawnImg = []
    for i in range(0,28):
        for j in range(0,28):
            drawnImg.append([float(jsonObject[str(i)][str(j)])/255])
    
    print(drawnImg)
    return drawnImg

@app.route('/prediction')
def prediction():
    data = request.args.get('jsdata')
    data = unquote_plus(data)
    
    if data:
        prediction = model.predict(transformData(data))
        
        text = "Prediction: "+str(prediction)
        return render_template('prediction.html',prediction=text,error="")
    else:
        return render_template('prediction.html',prediction="Error",error="error")

if __name__ == '__main__':
    app.run(debug=True)