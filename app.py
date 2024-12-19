import joblib
import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load models
lda = joblib.load('D:\Environments\Projects\Wheat-Kernel-Classification\experiments\lda_transformer.joblib')
model = pickle.load(open('D:\Environments\Projects\Wheat-Kernel-Classification\experiments\clf.pkl', 'rb'))

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
        attr = np.array(inputs).reshape(1, -1)
        scaled_attr = lda.transform(attr)

        # Make prediction
        predictions = model.predict(scaled_attr)
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
    app.run(debug=True)