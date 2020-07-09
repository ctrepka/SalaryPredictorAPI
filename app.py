import numpy as np
from flask import Flask, request, jsonify, render_template
import os
from os import path
import pickle
import multiple_linear_regression

#initialize flask app
app = Flask(__name__)

#check if model already exists on directory, generate pkl if it does not
if not path.exists('model.pkl'):
    print('creating model.pkl...')
    multiple_linear_regression.create_pkl()
    print('pkl created...')

#loading model.pkl as bytestream
model = pickle.load(open('model.pkl', 'rb'))
print('pkl loaded...')

#making index route for flask app
@app.route('/')
def home():
    return render_template('index.html')

#creating route for POST operation to complete prediction on submitted values
@app.route('/predict', methods=['POST'])
def predict():
    '''For rendering results on HTML GUI'''
    job = False

    if request.values['job'] == 'teacher':
        job = [0.0, 1.0]
    elif request.values['job'] == 'coder':
        job = [1.0, 0.0]
    else:
        return render_template('index.html', prediction_text="Invalid input, please select from the options available")    

    
    values = job + [float(request.values['xp'])]
    final_features = [np.array(values)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text="Salary will be around $ {}".format(output))    


if __name__ == "__main__":
        app.run(debug=True)