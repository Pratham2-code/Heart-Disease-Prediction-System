# app.py

import joblib
import numpy as np
from flask import Flask, request, render_template

# Initialize Flask App
app = Flask(__name__)

# --- Load the Model ---
# The model must be loaded once when the application starts.
# We are assuming 'model.joblib' contains the trained Random Forest Classifier.
try:
    model = joblib.load('model.joblib')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Define the features exactly as they appear in the CSV/training data
FEATURE_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
    'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

# --- Flask Routes ---

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = None
    
    if request.method == 'POST':
        if model is None:
            prediction_text = "ERROR: Prediction model is not loaded. Check server logs."
            return render_template('index.html', prediction_text=prediction_text)

        try:
            # 1. Collect all 13 features from the submitted form
            features = []
            for name in FEATURE_NAMES:
                # Convert form data (strings) to float/int
                value = float(request.form[name])
                features.append(value)
            
            # 2. Convert features into a NumPy array, reshaping for the model
            # The model expects a 2D array: [[feature_1, feature_2, ..., feature_13]]
            input_data = np.array([features])
            
            # 3. Make Prediction
            # Note: If the model was trained on SCALED data, this step would require loading and using the StandardScaler object here.
            prediction = model.predict(input_data)[0]
            
            # 4. Format the result
            if prediction == 1:
                prediction_text = "The model predicts: HIGH likelihood of Heart Disease. Please consult a medical professional."
            else:
                prediction_text = "The model predicts: LOW likelihood of Heart Disease. Always consult a medical professional for diagnosis."
                
        except ValueError:
            prediction_text = "ERROR: Please ensure all 13 fields are filled with valid numeric values."
        except Exception as e:
            prediction_text = f"An unexpected error occurred during prediction: {e}"

    # Render the HTML form with the prediction result (or None on GET request)
    return render_template('index.html', prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)