from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler



application = Flask(__name__)
app = application

## import ridge resgressor and standard scaler pickle
ridge_model = pickle.load(open('models/ridge.pkl','rb'))

standard_scaler = pickle.load(open('models/scaler.pkl','rb'))




@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'POST':
        temperature = float(request.form['Temperature'])
        RH = float(request.form['RH'])
        Ws = float(request.form['Ws'])
        Rain = float(request.form['Rain'])
        FFMC = float(request.form['FFMC'])
        DMC = float(request.form['DMC'])
        ISI = float(request.form['ISI'])
        Classes = float(request.form['Classes'])
        Region = float(request.form['Region'])
        

        scaled_data = standard_scaler.transform([[temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]])
        prediction = ridge_model.predict(scaled_data)

        return render_template('home.html', result=prediction[0])
    else:
        return render_template('home.html')



if __name__ == '__main__':
    app.run(host="0.0.0.0", port = 8080)
    
    
    #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