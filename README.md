# 🌲 Forest Fire Weather Index (FWI) Prediction

A machine learning web application that predicts the **Fire Weather Index (FWI)** from meteorological inputs, built on the Algerian Forest Fires dataset and deployed via a Flask web interface.

---

## Overview

Forest fires are among the most devastating natural disasters. Early risk assessment is critical for timely evacuation and resource allocation. This project trains a **Linear Regression** model to predict the FWI — a globally recognised numeric rating of fire intensity — and serves predictions through an interactive web application.

The model was trained on data from two Algerian regions (Bejaia and Sidi Bel-Abbes) covering June–September 2012.

---

## Dataset

- **Source:** [Algerian Forest Fires Dataset — UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/547/algerian+forest+fires+dataset)
- **Instances:** 243 (122 from Bejaia, 121 from Sidi Bel-Abbes)
- **Time period:** June 2012 – September 2012
- **Features:** 14 attributes — day, month, Temperature, RH, Ws, Rain, FFMC, DMC, DC, ISI, BUI, FWI, Classes, Region
- **Target variable:** `FWI` (continuous numeric value)

---

## Model Performance

| Model | MAE | R² Score |
|---|---|---|
| **Linear Regression** | **0.5796** | **0.9838** |
| Random Forest Regressor | 0.7456 | 0.9723 |

Linear Regression was selected for deployment due to its superior performance and interpretability on this dataset.

---

## Methodology

1. **Data Cleaning & Encoding** — Encode `Classes` (fire/not fire) to binary (1/0); drop the constant `year` column
2. **Exploratory Data Analysis** — Correlation heatmap to understand feature relationships
3. **Feature Selection** — Remove features with Pearson correlation > 0.85 (dropped `DC` and `BUI`); reduces features from 13 → 11
4. **Train-Test Split** — 75% training (182 samples) / 25% testing (61 samples)
5. **Feature Scaling** — `StandardScaler` applied to training data; same fitted scaler used on test data to prevent leakage
6. **Model Training** — Linear Regression and Random Forest Regressor trained and compared
7. **Deployment** — Flask web app with pickle-serialised model and scaler for real-time predictions

---

## 🛠️ Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3 |
| Data Manipulation | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Model Serialisation | Pickle |
| Web Framework | Flask |
| Frontend | HTML5, CSS3, JavaScript |
| Deployment | Flask Development Server (port 8080) |

---

## Project Structure

```
forestfires/
│
├── models/
│   ├── linear_model.pkl       # Serialised trained model
│   └── scaler.pkl             # Serialised StandardScaler
│
├── templates/
│   └── home.html              # Flask HTML template
│
├── static/                    # CSS / JS assets
│
├── Algerian_forest_fires_dataset.csv
├── notebook.ipynb             # Training and EDA notebook
├── app.py                     # Flask application
└── requirements.txt
```

---

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
python app.py
```

Navigate to `http://localhost:8080` in your browser.

### Input Features

Enter the following meteorological and fire-index values in the web form:

| Feature | Description |
|---|---|
| Day | Day of the month |
| Month | Month (numeric) |
| Temperature | Temperature in °C |
| RH | Relative Humidity (%) |
| Ws | Wind Speed (km/h) |
| Rain | Rainfall (mm) |
| FFMC | Fine Fuel Moisture Code |
| DMC | Duff Moisture Code |
| ISI | Initial Spread Index |
| Classes | Fire / Not Fire (1 / 0) |
| Region | Region code |

---

## Future Scope

- **Expanded Dataset** — Include more regions and time periods for better generalisation
- **Advanced Models** — Explore XGBoost, LightGBM, and LSTM for temporal patterns
- **Hyperparameter Tuning** — GridSearchCV or Bayesian optimisation
- **Real-Time Data** — Integrate live weather APIs (e.g., OpenWeatherMap)
- **Classification Module** — Map predicted FWI to fire danger categories (Low → Extreme)
- **Cloud Deployment** — Deploy on AWS Elastic Beanstalk, Heroku, or Google Cloud Run
- **Mobile App** — Companion app for field officers and forest rangers
- **Explainability** — Integrate SHAP or LIME for feature importance insights

---
