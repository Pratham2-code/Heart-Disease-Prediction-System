# HeartPulse AI: Radium Heart Disease Prediction & XAI System

HeartPulse AI is a modern, responsive web application that predicts the likelihood of heart disease using Machine Learning and interprets those predictions using **Explainable AI (XAI)** techniques. 

Built on a **Flask** backend with a premium, futuristic glassmorphic radium-cyan user interface, the system leverages **SHAP** and **LIME** to explain *why* the model made a specific classification—providing transparent, feature-level insights into patient risk factors.

---

## 🚀 Key Features

* **Predictive Power:** Employs a pre-trained machine learning classifier to estimate heart disease risk.
* **Explainable AI (XAI):**
  * **LIME (Local Interpretable Model-agnostic Explanations):** Explains individual predictions by identifying local feature contributions.
  * **SHAP (SHapley Additive exPlanations):** Calculates Shapley values to measure the global and local impact of each clinical attribute.
* **Neural Insights:** Highlights the top risk factors and healthy indicators for each prediction.
* **Radium Dark-Mode UI:** A high-end, responsive, glassmorphic dashboard styled with modern typography and sleek cybernetic accents.
* **Cloud Ready:** Fully configured with `Procfile` and `runtime.txt` for deployment on Render or Railway.

---

## 📁 Repository Structure

```text
├── app.py                      # Flask web application & XAI interface
├── model.joblib                # Serialized trained machine learning model
├── scaler.joblib               # Serialized StandardScaler object
├── heart_cleveland_upload.csv  # Cleveland Heart Disease Dataset (background distribution)
├── Procfile                    # Gunicorn startup command (--timeout 120 --workers 1)
├── runtime.txt                 # Specifies Python version (python-3.12.9)
├── requirements.txt            # Python dependencies
├── .gitignore                  # Excludes cache, training script & notebooks from Git
└── templates/
    └── index.html              # Frontend dashboard with interactive prediction forms & charts
```

> **Note:** `model.py` (the training script) is excluded from the repository via `.gitignore` as the pre-trained artifacts (`model.joblib`, `scaler.joblib`) are already committed.

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

5. **Run the application locally:**
   ```bash
   python app.py
   ```
   Open your browser and navigate to `http://127.0.0.1:5000`.

---

## ⚡ Deployment Guide

This project is pre-configured with **Gunicorn** (`--timeout 120 --workers 1`) and standard deployment configurations. The `PORT` environment variable is automatically read from the host platform.

### Deploy to Render
1. Create a new **Web Service** on Render connected to your Git repository.
2. Select **Python** as the runtime environment.
3. Configure the commands:
   * **Build Command:** `pip install -r requirements.txt`
   * **Start Command:** Auto-detected from `Procfile` (`gunicorn app:app --timeout 120 --workers 1`)
4. Click **Deploy Web Service** — no additional environment variables required.

### Deploy to Railway
1. Start a new project on Railway and choose **Deploy from GitHub**.
2. Select this repository.
3. Railway will auto-detect the `runtime.txt` and `Procfile` and deploy your app instantly.

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
