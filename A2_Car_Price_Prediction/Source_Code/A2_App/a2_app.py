from flask import Flask, render_template, request
from datetime import datetime
import a1_predict_car_price as a1_pred
from a2_predict_car_price import *;

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/a1_load')
def a1_load():
    return render_template('a1_index.html')

@app.route('/a2_load')
def a2_load():
    return render_template('a2_index.html')

@app.route('/a1_predict',methods=['POST'])
def a1_predict():
    input_features = [float(request.form['max_power']),
                      (datetime.now().year) - float(request.form['car_age']),
                      float(request.form['mileage'])]
    prediction = a1_pred.fn_predict(input_features)
    return render_template('a1_index.html',prediction=prediction)

@app.route('/a2_predict',methods=['POST'])
def a2_predict():
    a2_input_features = [float(request.form['max_power']),
                      (datetime.now().year) - float(request.form['car_age']),
                      float(request.form['mileage'])]
    prediction = fn_a2_predict(a2_input_features)
    return render_template('a2_index.html',prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=1986)