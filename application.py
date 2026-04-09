from flask import Flask, flash, get_flashed_messages, redirect, render_template, request, url_for
import pickle
import pandas as pd

application = Flask(__name__)
app = application
app.secret_key = "fwi-prediction-secret-key"

## import linear regression model 
model = pickle.load(open('models/linear_model.pkl','rb'))
standard_scaler = pickle.load(open('models/scaler.pkl','rb'))


@app.route('/')
def index():
    error_message = None
    flashed_messages = get_flashed_messages(with_categories=True)
    for category, message in flashed_messages:
        if category == 'error':
            error_message = message
            break
    return render_template('home.html', error=error_message)

@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return redirect(url_for('index'))
    try:
        day = int(request.form['Day'])
        month = int(request.form['Month'])
        temperature = float(request.form['Temperature'])
        RH = float(request.form['RH'])
        Ws = float(request.form['Ws'])
        Rain = float(request.form['Rain'])
        FFMC = float(request.form['FFMC'])
        DMC = float(request.form['DMC'])
        ISI = float(request.form['ISI'])
        Classes = float(request.form['Classes'])
        Region = float(request.form['Region'])

        input_row = pd.DataFrame([ 
            {
                'day': day,
                'month': month,
                'Temperature': temperature,
                'RH': RH,
                'Ws': Ws,
                'Rain': Rain,
                'FFMC': FFMC,
                'DMC': DMC,
                'ISI': ISI,
                'Classes': Classes,
                'Region': Region,
            }
        ])

        scaled_data = standard_scaler.transform(input_row)
        prediction = model.predict(scaled_data)
        return render_template('home.html', result=round(float(prediction[0]), 3))
    except (ValueError, KeyError):
        flash('Please enter valid numeric values for all fields.', 'error')
        return redirect(url_for('index'))



if __name__ == '__main__':
    app.run(host="0.0.0.0", port = 8080)
    
    
    
