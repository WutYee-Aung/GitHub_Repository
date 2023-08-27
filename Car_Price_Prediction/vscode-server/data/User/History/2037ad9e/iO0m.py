from flask import Flask, render_template, request
import source_code.predict_car_price as predict_car_price

# app = Flask(__name__)

# @app.route('/')
# def hello_world():
    # return "Hello World"

# if __name__ == '__main__':
    # app.run()


# app = Flask(__name__)

# @app.route('/',methods=['Get'])
# def hello_world():
#     return render_template('index.html')

# if __name__ == '__main__':
#     app.run()


app = Flask(__name__)

def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    input_features = [float(request.form['max_power']),
                      float(request.form['car_age']),
                      float(request.form['mileage']),]
    prediction = predict_car_price.fn_predict(input_features)
    return render_template('index.html',prediction)

if __name__ == '__main__':
    app.run()