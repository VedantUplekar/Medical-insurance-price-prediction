import pickle
import numpy as np
from flask import Flask, request, jsonify

from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Your prediction logic here
    return jsonify({'prediction': 'some_value'})

if __name__ == '__main__':
    app.run(debug=True)

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
with  open(r"C://Users//VEDANT UPLEKAR//Downloads//project 6//insurancemodelf.pkl", "rb") as file:
    model = pickle.load(file)

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
        print(prediction)

        # Return the predicted cost
        return jsonify({'predicted_cost': f'{round(float(prediction[0]), 2)} rs'})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
