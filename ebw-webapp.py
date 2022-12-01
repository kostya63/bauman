from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn import preprocessing
import joblib


app = Flask(__name__)
dnn_model = keras.models.load_model('ebw-dnn-model.h5', compile = True)
RFR_model = joblib.load('RFR_model.sav')
GBR_model = joblib.load('GBR_model.sav')
BR_model = joblib.load('BR_model.sav')
scaler = joblib.load('scaler.sav')


def get_predictions(IW, IF, VW, FP):
    X1 = pd.DataFrame(data = np.array([IW, IF, VW, FP], ndmin=2), columns = ['IW', 'IF', 'VW', 'FP'])
    dnn_y1 = dnn_model.predict(X1)
    X1_scaled = pd.DataFrame(data = scaler.transform(X1), columns = ['IW', 'IF', 'VW', 'FP'])
    y1 = RFR_model.predict(X1_scaled)
    y2 = GBR_model.predict(X1_scaled)
    y3 = BR_model.predict(X1_scaled)
    
    message1 = 'Deep Neural Network prediction: ' + f'Depth = {str(round(dnn_y1[0, 0], 2))} ' + f'Width = {str(round(dnn_y1[0, 1], 2))}'
    message2 = 'Random Forest prediction:       ' + f'Depth = {round(y1[0, 0], 2)} ' + f'Width = {round(y1[0, 1], 2)}'
    message3 = 'GradientBoosting prediction:    ' + f'Depth = {round(y2[0, 0], 2)} ' + f'Width = {round(y2[0, 1], 2)}'
    message4 = 'BaggingRegressor prediction:    ' + f'Depth = {round(y3[0, 0], 2)} ' + f'Width = {round(y3[0, 1], 2)}'
    return message1 + '\n' + message2 + '\n' + message3 + '\n' + message4


@app.route('/')
def index():
    return "main"


@app.route('/predict/', methods=['post', 'get'])
def processing():
    prediction = ''
    if request.method == 'POST':
        IW = float(request.form.get('IW'))
        IF = float(request.form.get('IF'))
        VW = float(request.form.get('VW'))
        FP = float(request.form.get('FP'))
        prediction = get_predictions(IW, IF, VW, FP)

    return render_template('predict.html', prediction = prediction)


if __name__ == '__main__':
    app.run()

