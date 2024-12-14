from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import pickle

app = Flask(__name__)

# Load the pre-trained model
model_path = 'F:/study/HK1_3_2425/DS321_MachineLearning1/Breast_Cancer_Classification/model.pkl'
model = pickle.load(open(model_path, 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input features from the form
        features = [float(request.form[feature]) for feature in ['area_worst', 'concave_points_worst', 'concave_points_mean', 'radius_worst', 'perimeter_worst']]
        features = np.array(features).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)[0]
        result = 'Malignant' if prediction == 1 else 'Benign'
        
        # Render the result
        return render_template('index.html', result=f'Predicted: {result}')
    except Exception as e:
        return render_template('index.html', result=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)