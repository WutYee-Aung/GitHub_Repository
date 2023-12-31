from flask import Flask, render_template, request
from datetime import datetime
import predict_car_price
# import source_code.predict_car_price as predict_car_price

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    input_features = [float(request.form['max_power']),
                      (datetime.now().year) - float(request.form['car_age']),
                      float(request.form['mileage'])]
    prediction = predict_car_price.fn_predict(input_features)
    return render_template('index.html',prediction=prediction)

if __name__ == '__main__':
    app.run()