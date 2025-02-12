import pickle

import numpy as np
from flask import Flask, jsonify, render_template, request

with open('house_data.pkl', 'rb') as f:
    model = pickle.load(f)


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
   
    area = float(request.form['area'])
    
   
    prediction = model.predict(np.array([[area]]))[0]
    
    
    return jsonify({'predicted_price': round(prediction, 2)})

if __name__ == '__main__':
    app.run(debug=True)
