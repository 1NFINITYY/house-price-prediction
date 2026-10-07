"""
House Price Predictor — Streamlit Web Application
==================================================
Loads the trained Gradient Boosting model and predicts
house prices based on user-provided property details.
"""

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path


# ─────────────────────────────────────────────────────────
# Page configuration  (must be the very first Streamlit call)
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# ─────────────────────────────────────────────────────────
# Custom CSS — light, clean, professional styling
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main container */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 820px;
    }

    /* Header card */
    .header-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        color: white;
    }
    .header-card h1 {
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0 0 0.4rem 0;
        letter-spacing: -0.5px;
    }
    .header-card p {
        font-size: 1.05rem;
        color: #a0aec0;
        margin: 0;
    }

    /* Section labels */
    .section-label {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #4a5568;
        margin-bottom: 0.5rem;
        margin-top: 0.25rem;
    }

    /* Section card */
    .section-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem 1.75rem 1rem;
        margin-bottom: 1.25rem;
    }
    .section-card h3 {
        font-size: 1rem;
        font-weight: 600;
        color: #2d3748;
        margin: 0 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e2e8f0;
    }

    /* Prediction result */
    .price-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 1.75rem 2rem;
        text-align: center;
        color: white;
        margin: 1.25rem 0;
    }
    .price-label {
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        opacity: 0.85;
        margin-bottom: 0.5rem;
    }
    .price-value {
        font-size: 2.8rem;
        font-weight: 800;
        letter-spacing: -1px;
        line-height: 1.1;
    }
    .price-full {
        font-size: 0.95rem;
        opacity: 0.8;
        margin-top: 0.4rem;
    }

    /* Info badge */
    .badge {
        display: inline-block;
        background: #ebf8ff;
        color: #2b6cb0;
        border-radius: 20px;
        padding: 0.2rem 0.75rem;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 0.4rem;
    }

    /* Model info table */
    .metric-row {
        display: flex;
        justify-content: space-between;
        padding: 0.5rem 0;
        border-bottom: 1px solid #e2e8f0;
        font-size: 0.9rem;
    }
    .metric-row:last-child { border-bottom: none; }
    .metric-key { color: #4a5568; font-weight: 500; }
    .metric-val { color: #1a202c; font-weight: 700; }

    /* Predict button override */
    div[data-testid="stButton"] > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        cursor: pointer;
        transition: opacity 0.2s;
    }
    div[data-testid="stButton"] > button:hover {
        opacity: 0.88;
    }

    /* Disclaimer */
    .disclaimer {
        font-size: 0.8rem;
        color: #718096;
        text-align: center;
        margin-top: 0.75rem;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# Load the trained model  (with graceful error handling)
# ─────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "house_price_model.pkl"

@st.cache_resource
def load_model(path: Path):
    """Load and cache the trained pipeline from disk."""
    return joblib.load(path)

try:
    model = load_model(MODEL_PATH)
except FileNotFoundError:
    st.error(
        "⚠️ **Unable to load the trained model.**\n\n"
        f"Expected file: `{MODEL_PATH}`\n\n"
        "Please ensure `house_price_model.pkl` exists in the `models/` directory."
    )
    st.stop()
except Exception as exc:
    st.error(f"⚠️ **Unexpected error while loading the model:** `{exc}`")
    st.stop()


# ─────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────
st.markdown("""
<div class="header-card">
    <h1>🏠 House Price Predictor</h1>
    <p>Predict the estimated price of a house using machine learning.</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# Input Section
# ─────────────────────────────────────────────────────────
st.markdown("### 📋 House Details")

# ── Property dimensions ──────────────────────────────────
with st.container():
    st.markdown('<div class="section-card"><h3>🏗️ Property Dimensions</h3>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        area = st.number_input(
            "Area (sq ft)",
            min_value=500,
            max_value=20_000,
            value=6_000,
            step=100,
            help="Total built-up area in square feet"
        )
    with col2:
        bedrooms = st.number_input(
            "Bedrooms",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
            help="Number of bedrooms"
        )
    with col3:
        bathrooms = st.number_input(
            "Bathrooms",
            min_value=1,
            max_value=10,
            value=2,
            step=1,
            help="Number of bathrooms"
        )

    col4, col5 = st.columns(2)
    with col4:
        stories = st.number_input(
            "Stories",
            min_value=1,
            max_value=10,
            value=2,
            step=1,
            help="Number of floors / stories"
        )
    with col5:
        parking = st.number_input(
            "Parking Spaces",
            min_value=0,
            max_value=10,
            value=2,
            step=1,
            help="Number of covered parking spots"
        )

    st.markdown('</div>', unsafe_allow_html=True)

# ── Property features ────────────────────────────────────
with st.container():
    st.markdown('<div class="section-card"><h3>✅ Property Features</h3>', unsafe_allow_html=True)
    col6, col7, col8 = st.columns(3)

    with col6:
        mainroad = st.selectbox(
            "Main Road Access",
            options=["yes", "no"],
            help="Is the property on or near a main road?"
        )
        guestroom = st.selectbox(
            "Guest Room",
            options=["yes", "no"],
            help="Does the property have a separate guest room?"
        )
    with col7:
        basement = st.selectbox(
            "Basement",
            options=["yes", "no"],
            help="Does the property have a basement?"
        )
        hotwaterheating = st.selectbox(
            "Hot Water Heating",
            options=["yes", "no"],
            help="Is there a hot water heating system?"
        )
    with col8:
        airconditioning = st.selectbox(
            "Air Conditioning",
            options=["yes", "no"],
            help="Is air conditioning installed?"
        )
        prefarea = st.selectbox(
            "Preferred Area",
            options=["yes", "no"],
            help="Is the property in a preferred / premium locality?"
        )

    furnishingstatus = st.selectbox(
        "Furnishing Status",
        options=["furnished", "semi-furnished", "unfurnished"],
        help="Current furnishing status of the property"
    )

    st.markdown('</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────
st.markdown("---")

if st.button("🔍 Predict House Price"):

    # Build input DataFrame matching training column order
    input_data = pd.DataFrame({
        "area":             [area],
        "bedrooms":         [bedrooms],
        "bathrooms":        [bathrooms],
        "stories":          [stories],
        "mainroad":         [mainroad],
        "guestroom":        [guestroom],
        "basement":         [basement],
        "hotwaterheating":  [hotwaterheating],
        "airconditioning":  [airconditioning],
        "parking":          [parking],
        "prefarea":         [prefarea],
        "furnishingstatus": [furnishingstatus],
    })

    predicted_price = model.predict(input_data)[0]

    # Convert to Lakh (1 Lakh = 100,000)
    price_in_lakh = predicted_price / 100_000

    # Display result card
    st.markdown(f"""
    <div class="price-card">
        <div class="price-label">Estimated House Price</div>
        <div class="price-value">₹{price_in_lakh:.2f} Lakh</div>
        <div class="price-full">Full value: ₹{predicted_price:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        '<p class="disclaimer">⚠️ Prediction is an estimate based on the training dataset '
        'and should not be considered a professional property valuation.</p>',
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────
# About the Model
# ─────────────────────────────────────────────────────────
st.markdown("---")

with st.expander("ℹ️ About the Model", expanded=False):
    st.markdown("""
    **Model:** Gradient Boosting Regressor (`sklearn.ensemble.GradientBoostingRegressor`)

    **Task:** House Price Regression (supervised learning)

    **Training Data:** 545 housing records from the Housing dataset

    #### Evaluation Metrics (Test Set)
    | Metric | Value |
    |--------|------:|
    | MAE (Mean Absolute Error) | ₹9,67,227 |
    | RMSE (Root Mean Squared Error) | ₹12,99,730 |
    | R² Score | 0.6658 |

    #### All Models Compared
    | Model | MAE | RMSE | R² |
    |-------|----:|-----:|---:|
    | Linear Regression | ₹9,70,043 | ₹13,24,507 | 0.6529 |
    | Random Forest | ₹10,15,097 | ₹13,98,658 | 0.6130 |
    | **Gradient Boosting** ✓ | **₹9,67,227** | **₹12,99,730** | **0.6658** |

    > **Why Gradient Boosting?** It achieved the lowest MAE and RMSE, and the highest R² score
    among all models tested — making it the most accurate predictor on the test set.

    #### Hyperparameter Tuning (GridSearchCV)
    Best parameters found for Gradient Boosting:
    - `learning_rate` = 0.05
    - `max_depth` = 2
    - `n_estimators` = 100

    #### Top Features by Importance
    1. 🏡 **Area** — strongest predictor (~46%)
    2. 🚿 **Bathrooms** (~17%)
    3. ❄️ **Air Conditioning** (~9%)
    4. 🚗 **Parking** (~5%)
    5. 🏢 **Stories** (~5%)
    """)