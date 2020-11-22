import numpy as np
import pandas as pd
import pickle
from flask import Flask, request, render_template, url_for

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('Home.html')
@app.route('/about')
def about():
    return render_template('About.html')

@app.route('/prediction')
def prediction():
    return render_template('Heart_Diseaase_prediction.html')


@app.route('/predict',methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]

    features_name = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
       'exang', 'oldpeak', 'slope', 'ca', 'thal']

    df = pd.DataFrame(features_value, columns=features_name)
    output = model.predict(df)

    if output == 0:
        res_val = "No heart Disease"
    else:
        res_val = "Heart Disease"

    return render_template('Heart_Diseaase_prediction.html', prediction_text='Patient has {}'.format(res_val))

if __name__ == "__main__":
    app.run()
