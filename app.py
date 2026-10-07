import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# -----------------------------
# Load model
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent
model_path = BASE_DIR / "models" / "house_price_model.pkl"

model = joblib.load(model_path)


# -----------------------------
# Page configuration
# -----------------------------

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

st.title("🏠 House Price Predictor")
st.write("Enter the details of the house to estimate its price.")


# -----------------------------
# Input fields
# -----------------------------

area = st.number_input(
    "Area (sq ft)",
    min_value=500,
    max_value=20000,
    value=6000
)

bedrooms = st.number_input(
    "Bedrooms",
    min_value=1,
    max_value=10,
    value=3
)

bathrooms = st.number_input(
    "Bathrooms",
    min_value=1,
    max_value=10,
    value=2
)

stories = st.number_input(
    "Stories",
    min_value=1,
    max_value=10,
    value=2
)

parking = st.number_input(
    "Parking spaces",
    min_value=0,
    max_value=10,
    value=2
)

mainroad = st.selectbox(
    "Main road",
    ["yes", "no"]
)

guestroom = st.selectbox(
    "Guestroom",
    ["yes", "no"]
)

basement = st.selectbox(
    "Basement",
    ["yes", "no"]
)

hotwaterheating = st.selectbox(
    "Hot water heating",
    ["yes", "no"]
)

airconditioning = st.selectbox(
    "Air conditioning",
    ["yes", "no"]
)

prefarea = st.selectbox(
    "Preferred area",
    ["yes", "no"]
)

furnishingstatus = st.selectbox(
    "Furnishing status",
    ["furnished", "semi-furnished", "unfurnished"]
)


# -----------------------------
# Prediction
# -----------------------------

if st.button("Predict House Price"):

    new_house = pd.DataFrame({
        "area": [area],
        "bedrooms": [bedrooms],
        "bathrooms": [bathrooms],
        "stories": [stories],
        "mainroad": [mainroad],
        "guestroom": [guestroom],
        "basement": [basement],
        "hotwaterheating": [hotwaterheating],
        "airconditioning": [airconditioning],
        "parking": [parking],
        "prefarea": [prefarea],
        "furnishingstatus": [furnishingstatus]
    })

    predicted_price = model.predict(new_house)[0]

    st.success(
        f"Estimated House Price: ₹{predicted_price:,.2f}"
    )