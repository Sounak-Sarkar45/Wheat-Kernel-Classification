import joblib
import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load models
model = pickle.load(open('\home\ubuntu\experiments\svc.pkl', 'rb'))
# model = pickle.load(open('D:/Environments/Projects/Wheat-Kernel-Classification/experiments/svc.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    For rendering results on HTML GUI
    """
    try:
        # Extract input values from form
        inputs = [
            float(request.form['area']),
            float(request.form['perimeter']),
            float(request.form['compactness']),
            float(request.form['kernel_length']),
            float(request.form['kernel_width']),
            float(request.form['coeff']),
            float(request.form['kernel_groove_length'])
        ]
        # Reshape and transform input
        attr = np.array([inputs])

        # Make prediction
        predictions = model.predict(attr)
        if predictions[0] == 1:
            output = "Kama"
        elif predictions[0] == 2:
            output = "Rosa"
        elif predictions[0] == 3:
            output = "Canadian"
        else:
            output = "Unknown"

        return render_template('index.html', prediction_text=f'Predicted Wheat Kernel: {output}')

    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')


if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8080)
    # app.run(debug=True)