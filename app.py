import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
sc = StandardScaler()
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    float_features = [float(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    Scaled_data    = sc.fit_transform(final_features)
    prediction = model.predict(Scaled_data) 

    if prediction == 1:
        pred = "Congrtas!, Customer is eligible for House Loan"
    elif prediction == 0:
        pred = "Sorry, Customer is Not eligible.Better to try next time"
    output = pred

    return render_template('result.html', prediction_text='{}'.format(output))
    
    
@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=[float(x) for x in request.json['data'].values()]
    val = data
    print(val)
    res = model.predict([val])
    if res == 1:
        pred = "Congrtas!, Customer is eligible for House Loan"
    elif res == 0:
        pred = "Sorry, Customer is Not eligible.Better to try next time"
    output = pred

    print(res)
    return jsonify({"success":True,
                                    "message":pred})

if __name__ == "__main__":
    app.run(debug=True)