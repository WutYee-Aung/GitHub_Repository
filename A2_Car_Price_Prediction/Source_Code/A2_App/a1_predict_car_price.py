import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import matplotlib
import pickle

# filename = '/root/source_code/a2_app/car_price_prediction.model'
filename = '/root/source_code/a2_app/a1_car_price_prediction.model'

def fn_predict(to_predict_list):

    to_predict = np.array(to_predict_list).reshape(1,3)

    loaded_model = pickle.load(open(filename, 'rb'))
    model = loaded_model['model']
    scaler = loaded_model['scaler']

    to_predict = scaler.transform(to_predict)

    result = model.predict(to_predict)
    return np.exp(result[0])