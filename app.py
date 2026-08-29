# app.py

import os
import joblib
import numpy as np
import pandas as pd
import shap
import lime
import lime.lime_tabular
from flask import Flask, request, render_template

# Initialize Flask App
app = Flask(__name__)

# --- Load the Model ---
# The model must be loaded once when the application starts.
# We are assuming 'model.joblib' contains the trained Random Forest Classifier.
try:
    model = joblib.load('model.joblib')
    scaler = joblib.load('scaler.joblib')
    print("Model and Scaler loaded successfully!")
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    model = None
    scaler = None

# Define the features exactly as they appear in the CSV/training data
FEATURE_NAMES = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
    'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

# Mapping short feature names to human-readable labels
FEATURE_LABELS = {
    'age': 'Age',
    'sex': 'Gender',
    'cp': 'Chest Pain Type',
    'trestbps': 'Resting Blood Pressure',
    'chol': 'Cholesterol Level',
    'fbs': 'Fasting Blood Sugar',
    'restecg': 'Resting ECG Results',
    'thalach': 'Max Heart Rate',
    'exang': 'Exercise Angina',
    'oldpeak': 'ST Depression',
    'slope': 'ST Slope',
    'ca': 'Major Vessels',
    'thal': 'Thalassemia'
}


def _extract_positive_class_shap(shap_output):
    """Return the SHAP contributions for the positive-risk class (class 1) in the same feature order used by the model."""
    try:
        values = np.asarray(shap_output.values)
    except Exception:
        return np.zeros(len(FEATURE_NAMES), dtype=float)

    if values.size == 0:
        return np.zeros(len(FEATURE_NAMES), dtype=float)

    arr = values
    if arr.ndim == 3:
        if arr.shape[-1] == 2:
            arr = arr[0, :, 1]
        else:
            arr = arr[0, :, 0]
    elif arr.ndim == 2:
        if arr.shape[1] == 2:
            arr = arr[:, 1]
        elif arr.shape[0] == len(FEATURE_NAMES):
            arr = arr[:, 0]
        else:
            arr = arr[0]
    elif arr.ndim == 1:
        arr = arr
    else:
        arr = np.zeros(len(FEATURE_NAMES), dtype=float)

    arr = np.asarray(arr, dtype=float).reshape(-1)
    if arr.size < len(FEATURE_NAMES):
        padded = np.zeros(len(FEATURE_NAMES), dtype=float)
        padded[:arr.size] = arr
        return padded
    return arr[:len(FEATURE_NAMES)]


# --- Initialize XAI Explainers ---
shap_explainer = None
lime_explainer = None

try:
    if model and scaler:
        # Load background data for XAI initialization
        df_bg = pd.read_csv('heart_cleveland_upload.csv')
        # Ensure we drop the target column safely
        target_col = 'condition' if 'condition' in df_bg.columns else 'target'
        X_train_raw = df_bg.drop(target_col, axis=1)
        X_train_scaled = scaler.transform(X_train_raw)
        
        # Initialize SHAP
        shap_explainer = shap.Explainer(model, X_train_scaled)
        
        # Initialize LIME
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train_scaled,
            feature_names=FEATURE_NAMES,
            class_names=['Normal', 'Risk'],
            mode='classification'
        )
        print("XAI Explainers (SHAP & LIME) initialized successfully!")
except Exception as e:
    print(f"Error initializing XAI: {e}")

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
            
            # 2. Convert features into a NumPy array
            input_raw = np.array([features])
            
            # 3. Scale the input data using a DataFrame to maintain feature names
            input_scaled = input_raw
            if scaler:
                input_df = pd.DataFrame(input_raw, columns=FEATURE_NAMES)
                input_scaled = scaler.transform(input_df)
            
            # 4. Make Prediction
            prediction = model.predict(input_scaled)[0]
            
            # 5. XAI Analysis
            xai_data = {'shap': {}, 'lime': {}}

            if shap_explainer:
                shap_values = shap_explainer(input_scaled)
                positive_class_values = _extract_positive_class_shap(shap_values)
                for i, name in enumerate(FEATURE_NAMES):
                    xai_data['shap'][name] = float(positive_class_values[i])

            xai_data['lime'] = {name: 0.0 for name in FEATURE_NAMES}
            if lime_explainer:
                # Explain the positive class (risk) to match the model's probability output.
                exp = lime_explainer.explain_instance(
                    input_scaled[0],
                    model.predict_proba,
                    num_features=len(FEATURE_NAMES),
                    labels=[1]
                )
                if 1 in exp.local_exp:
                    for feature_idx, weight in exp.local_exp[1]:
                        if 0 <= feature_idx < len(FEATURE_NAMES):
                            xai_data['lime'][FEATURE_NAMES[feature_idx]] = float(weight)

            # 6. Generate Neural Insights for explanation
            insights = {
                'lime_top_risk': sorted([k for k, v in xai_data['lime'].items() if v > 0], key=lambda x: xai_data['lime'][x], reverse=True)[:2],
                'lime_top_healthy': sorted([k for k, v in xai_data['lime'].items() if v < 0], key=lambda x: xai_data['lime'][x])[:2],
                'shap_top_risk': sorted([k for k, v in xai_data['shap'].items() if v > 0], key=lambda x: xai_data['shap'][x], reverse=True)[:2],
                'shap_top_healthy': sorted([k for k, v in xai_data['shap'].items() if v < 0], key=lambda x: xai_data['shap'][x])[:2]
            }

            # 7. Format the result
            if prediction == 1:
                prediction_text = "The model predicts: HIGH likelihood of Heart Disease. Please consult a medical professional."
            else:
                prediction_text = "The model predicts: LOW likelihood of Heart Disease. Always consult a medical professional for diagnosis."
                
            return render_template('index.html', 
                                 prediction_text=prediction_text, 
                                 xai_data=xai_data,
                                 feature_names=FEATURE_NAMES,
                                 feature_labels=FEATURE_LABELS,
                                 insights=insights)

        except ValueError as ve:
            prediction_text = f"ERROR: Invalid input data. {ve}"
        except Exception as e:
            prediction_text = f"An unexpected error occurred during prediction: {e}"

    # Render the HTML form with the prediction result (or None on GET request)
    return render_template('index.html', prediction_text=prediction_text)

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)