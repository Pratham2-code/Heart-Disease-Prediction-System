# HeartPulse AI: Radium Heart Disease Prediction & XAI System

HeartPulse AI is a modern, responsive web application that predicts the likelihood of heart disease using Machine Learning and interprets those predictions using **Explainable AI (XAI)** techniques. 

Built on a **Flask** backend with a premium, futuristic glassmorphic radium-cyan user interface, the system leverages **SHAP** and **LIME** to explain *why* the model made a specific classification—providing transparent, feature-level insights into patient risk factors.

---

## 🚀 Key Features

* **Predictive Power:** Employs a pre-trained machine learning classifier (Logistic Regression/Random Forest) to estimate heart disease risk.
* **Explainable AI (XAI):**
  * **LIME (Local Interpretable Model-agnostic Explanations):** Explains individual predictions by identifying local feature contributions.
  * **SHAP (SHapley Additive exPlanations):** Calculates Shapley values to measure the global and local impact of each clinical attribute.
* **Neural Insights:** Highlights the top risk factors and healthy indicators for each prediction.
* **Radium Dark-Mode UI:** A high-end, responsive, glassmorphic dashboard styled with modern typography and sleek cybernetic accents.
* **Cloud Ready:** Fully configured with `Procfile` and `runtime.txt` for deployment on Render, Railway, or Heroku.

---

## 📁 Repository Structure

```text
├── app.py                      # Flask web application & XAI interface
├── model.py                    # Model training, evaluation, & serialization script
├── model.joblib                # Serialized trained machine learning model
├── scaler.joblib               # Serialized StandardScaler object
├── heart_cleveland_upload.csv  # Cleveland Heart Disease Dataset (background distribution)
├── Procfile                    # Deployment startup script for WSGI servers (Gunicorn)
├── runtime.txt                 # Specifies Python version for cloud deployment
├── requirements.txt            # Python dependencies
└── templates/
    └── index.html              # Frontend dashboard with interactive prediction forms & charts
```

---

## 🛠️ Installation & Local Setup

### Prerequisites
* Python 3.12+ (Recommended)
* Git

### Step-by-Step Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repository-url>
   cd heart-diseases-prediction
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   * **Windows (PowerShell):**
     ```powershell
     .\venv\Scripts\Activate.ps1
     ```
   * **macOS / Linux:**
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Train the model (Optional):**
   If you want to retrain the model and save a new `model.joblib` and `scaler.joblib`, run:
   ```bash
   python model.py
   ```

6. **Run the application locally:**
   ```bash
   python app.py
   ```
   Open your browser and navigate to `http://127.0.0.1:5001`.

---

## ⚡ Deployment Guide

This project is pre-configured with **Gunicorn** and standard deployment configurations.

### Deploy to Render
1. Create a new **Web Service** on Render connected to your Git repository.
2. Select **Python** as the runtime environment.
3. Configure the commands:
   * **Build Command:** `pip install -r requirements.txt`
   * **Start Command:** `gunicorn app:app` (automatically detected from the `Procfile`)
4. Add environment variables if needed, then click **Deploy Web Service**.

### Deploy to Railway
1. Start a new project on Railway and choose **Deploy from GitHub**.
2. Select this repository.
3. Railway will auto-detect the `runtime.txt` and `Procfile` and deploy your app instantly.

### Deploy to Heroku
1. Log in via Heroku CLI:
   ```bash
   heroku login
   ```
2. Create your Heroku app:
   ```bash
   heroku create heartpulse-ai
   ```
3. Push your repository to Heroku:
   ```bash
   git push heroku main
   ```

---

## 🔬 Clinical Feature Meanings

The model uses 13 clinical attributes to predict heart disease:
1. **Age:** Age in years.
2. **Gender:** Male (1) or Female (0).
3. **Chest Pain Type:** Typical angina (0), atypical angina (1), non-anginal pain (2), asymptomatic (3).
4. **Resting Blood Pressure:** mm Hg on admission.
5. **Cholesterol Level:** Serum cholesterol in mg/dl.
6. **Fasting Blood Sugar:** > 120 mg/dl (1 = true; 0 = false).
7. **Resting ECG Results:** Normal (0), ST-T wave abnormality (1), left ventricular hypertrophy (2).
8. **Max Heart Rate:** Maximum heart rate achieved during exercise.
9. **Exercise Angina:** Exercise-induced angina (1 = yes; 0 = no).
10. **ST Depression:** ST depression induced by exercise relative to rest.
11. **ST Slope:** Upsloping (0), flat (1), downsloping (2).
12. **Major Vessels:** Number of major vessels colored by fluoroscopy (0-3).
13. **Thalassemia:** Normal (3), fixed defect (6), reversible defect (7).

---

## 🛡️ License

This project is licensed under the MIT License.
