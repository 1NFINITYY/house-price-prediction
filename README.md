# 🏠 House Price Prediction

A beginner-level Machine Learning project that predicts house prices based on property characteristics using a **Gradient Boosting Regressor**. Includes exploratory data analysis (EDA), preprocessing, model comparison, hyperparameter tuning, and a live **Streamlit web application**.

---

## Overview

This project uses machine learning to predict house prices based on property characteristics such as area, bedrooms, bathrooms, stories, parking, location-related features, and furnishing status.

The full ML workflow — from data loading through model deployment — is documented in the Jupyter notebook and exposed through a clean Streamlit web interface where anyone can enter property details and get an instant price estimate.

---

## Features

- ✅ House price prediction via trained ML model
- 📊 Exploratory Data Analysis (EDA) with visualisations
- 🔧 Data preprocessing and categorical encoding (One-Hot Encoding)
- 🤖 Multiple ML models trained and compared
- 📏 Model evaluation using MAE, RMSE, and R²
- 🔎 Hyperparameter tuning with GridSearchCV
- 📈 Feature importance analysis
- 🌐 Interactive Streamlit web application

---

## Dataset

| Property | Value |
|----------|-------|
| Source | Housing.csv |
| Rows | 545 |
| Columns | 13 |
| Target Variable | `price` (house price in ₹) |

### Features

**Numerical:**
| Feature | Description |
|---------|-------------|
| `area` | Total built-up area (sq ft) |
| `bedrooms` | Number of bedrooms |
| `bathrooms` | Number of bathrooms |
| `stories` | Number of floors |
| `parking` | Number of parking spaces |

**Categorical:**
| Feature | Values |
|---------|--------|
| `mainroad` | yes / no |
| `guestroom` | yes / no |
| `basement` | yes / no |
| `hotwaterheating` | yes / no |
| `airconditioning` | yes / no |
| `prefarea` | yes / no |
| `furnishingstatus` | furnished / semi-furnished / unfurnished |

---

## Machine Learning Workflow

```
Dataset (Housing.csv)
        ↓
  Data Loading & Inspection
        ↓
  Exploratory Data Analysis
        ↓
  Data Preprocessing
  (One-Hot Encoding, ColumnTransformer)
        ↓
  Train / Test Split (80/20)
        ↓
  Model Training (3 models)
        ↓
  Model Evaluation (MAE, RMSE, R²)
        ↓
  Hyperparameter Tuning (GridSearchCV)
        ↓
  Model Saving (house_price_model.pkl)
        ↓
  Streamlit Deployment
```

---

## Exploratory Data Analysis

The notebook covers the following EDA steps:

- **Price distribution** — histogram of house prices to understand spread and skewness
- **Area vs Price** — scatter plot revealing a positive correlation between area and price
- **Correlation matrix** — heatmap of numerical feature correlations
- **Categorical feature analysis** — boxplots showing price variation across each categorical feature
- **Missing value check** — confirmed no missing values in the dataset
- **Data type inspection** — verified correct dtypes for all columns

**Key findings:**
- `area` is the strongest predictor of price
- Houses with `airconditioning`, in a `prefarea`, or on the `mainroad` tend to be priced higher
- `furnishingstatus` shows a visible price difference between furnished and unfurnished homes

---

## Models Tested

| Model | MAE | RMSE | R² |
|-------|----:|-----:|---:|
| Linear Regression | ₹9,70,043 | ₹13,24,507 | 0.6529 |
| Random Forest | ₹10,15,097 | ₹13,98,658 | 0.6130 |
| **Gradient Boosting** ✓ | **₹9,67,227** | **₹12,99,730** | **0.6658** |

---

## Model Selection

**Gradient Boosting Regressor** was selected as the final model because it achieved:
- The **lowest MAE** (₹9,67,227) — smallest average prediction error
- The **lowest RMSE** (₹12,99,730) — fewest large prediction errors
- The **highest R²** (0.6658) — explains ~66.6% of price variance

Linear Regression was a close competitor on MAE, but Gradient Boosting consistently outperformed on all three metrics.

---

## Feature Importance

Based on the Gradient Boosting model's feature importances (top 5):

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | Area | ~46% |
| 2 | Bathrooms | ~17% |
| 3 | Air Conditioning | ~9% |
| 4 | Parking | ~5% |
| 5 | Stories | ~5% |

`area` is by far the most influential feature, followed by `bathrooms` and `airconditioning`.

---

## Hyperparameter Tuning

GridSearchCV with 5-fold cross-validation was used to tune both Random Forest and Gradient Boosting.

**Best Gradient Boosting parameters found:**
```
learning_rate  = 0.05
max_depth      = 2
n_estimators   = 100
```

The saved model (`house_price_model.pkl`) uses the base Gradient Boosting configuration (before tuning), which scored higher on the held-out test set.

---

## Web Application

The Streamlit app (`app.py`) provides a clean, interactive interface to:

1. Enter property details (area, bedrooms, bathrooms, stories, parking)
2. Select property features (main road, guest room, basement, etc.)
3. Choose furnishing status
4. Click **Predict House Price** to get an instant estimate

The result is displayed prominently in ₹ Lakh along with the full value.

---

## Project Structure

```
housing_price_prediction/
│
├── data/
│   └── Housing.csv           ← Raw dataset
│
├── models/
│   └── house_price_model.pkl ← Trained Gradient Boosting pipeline
│
├── notebooks/
│   └── house_price_prediction.ipynb  ← Full EDA + training notebook
│
├── src/
│   └── predict.py            ← Command-line prediction script
│
├── app.py                    ← Streamlit web application
├── requirements.txt          ← Python dependencies
└── README.md                 ← This file
```

---

## Installation

**1. Clone the repository:**
```bash
git clone https://github.com/1NFINITYY/house-price-prediction.git
cd house-price-prediction
```

**2. Create a virtual environment:**
```bash
python -m venv .venv
```

**3. Activate the environment:**

Windows:
```bash
.venv\Scripts\activate
```

macOS / Linux:
```bash
source .venv/bin/activate
```

**4. Install dependencies:**
```bash
pip install -r requirements.txt
```

---

## Run the Streamlit App

```bash
streamlit run app.py
```

Streamlit will print a local URL (e.g. `http://localhost:8501`). Open it in your browser to use the app.

---

## Command-Line Prediction

To run an interactive command-line prediction:

```bash
python src/predict.py
```

You will be prompted to enter each property detail and the predicted price will be printed.

---

## Technologies Used

| Tool | Purpose |
|------|---------|
| **Python 3.x** | Core language |
| **Pandas** | Data manipulation |
| **NumPy** | Numerical operations |
| **Matplotlib** | Plotting (EDA) |
| **Seaborn** | Statistical visualisations |
| **Scikit-learn** | ML models, pipelines, preprocessing |
| **Joblib** | Model serialisation |
| **Streamlit** | Web application |
| **Jupyter Notebook** | Interactive development |

---

## Future Improvements

- [ ] Try **XGBoost** / **LightGBM** for potentially better performance
- [ ] Add more advanced **feature engineering** (e.g. price per sq ft)
- [ ] Collect a **larger dataset** for better generalisation
- [ ] Add **cross-validation** curves and learning curves
- [ ] Deploy to **Streamlit Cloud** or **Hugging Face Spaces**
- [ ] Add **model monitoring** for data drift
- [ ] Include **SHAP values** for better explainability
- [ ] Build a more advanced UI with filters and comparison mode

---

## Disclaimer

> Predictions generated by this model are **estimates** based on a small training dataset of 545 records. They should **not** be used as professional property valuations. Actual house prices depend on many additional factors not captured in this dataset.
