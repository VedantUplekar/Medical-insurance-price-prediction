import os
import numpy as np
from flask import Flask, request, jsonify

from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model

file_path = os.path.join(os.path.dirname(__file__), "insurancemodelf.pkl")
with open(file_path, "rb") as file:
    model = file.read()
    
    
@app.route('/')
def home():
    return "Medical Insurance Cost Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()

        # Extract features (ensure they match model training order)
        age = data['age']
        bmi = data['bmi']
        children = data['children']
        smoker = 1 if data['smoker'].lower() == 'yes' else 0
        features = np.array([[age, bmi, children, smoker]])


        
        prediction = model.predict(features)
        print('Prediction',prediction)

        # Return the predicted cost
        return jsonify({'predicted_cost': f'{round(float(prediction[0]), 2)} rs'})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
