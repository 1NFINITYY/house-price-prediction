"""
predict.py — Command-Line House Price Predictor
================================================
Run this script from the project root to interactively
predict a house price using the trained Gradient Boosting model.

Usage:
    python src/predict.py
"""

import joblib
import pandas as pd
from pathlib import Path


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

def get_numeric_input(prompt: str, min_val: float, max_val: float) -> float:
    """Prompt the user for a numeric value within an allowed range."""
    while True:
        try:
            value = float(input(prompt))
            if min_val <= value <= max_val:
                return value
            print(f"  ⚠  Please enter a value between {min_val} and {max_val}.")
        except ValueError:
            print("  ⚠  Invalid input. Please enter a number.")


def get_yes_no_input(prompt: str) -> str:
    """Prompt the user for a yes/no answer."""
    while True:
        answer = input(prompt).strip().lower()
        if answer in ("yes", "no"):
            return answer
        print("  ⚠  Please type 'yes' or 'no'.")


def get_furnishing_input() -> str:
    """Prompt the user for furnishing status."""
    options = ("furnished", "semi-furnished", "unfurnished")
    while True:
        answer = input(
            "Furnishing status (furnished / semi-furnished / unfurnished): "
        ).strip().lower()
        if answer in options:
            return answer
        print(f"  ⚠  Please enter one of: {', '.join(options)}")


# ─────────────────────────────────────────────────────────
# Load the trained model
# ─────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
model_path = BASE_DIR / "models" / "house_price_model.pkl"

try:
    model = joblib.load(model_path)
except FileNotFoundError:
    print(f"\n❌  Model file not found at: {model_path}")
    print("    Please ensure 'house_price_model.pkl' is in the models/ directory.\n")
    raise SystemExit(1)


# ─────────────────────────────────────────────────────────
# Collect user inputs
# ─────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("   House Price Predictor -- CLI")
print("=" * 50)
print("Enter the property details below.\n")

area             = get_numeric_input("Area (sq ft, 500–20000): ", 500, 20_000)
bedrooms         = int(get_numeric_input("Bedrooms (1–10): ", 1, 10))
bathrooms        = int(get_numeric_input("Bathrooms (1–10): ", 1, 10))
stories          = int(get_numeric_input("Stories (1–10): ", 1, 10))
mainroad         = get_yes_no_input("Main road access? (yes/no): ")
guestroom        = get_yes_no_input("Guest room? (yes/no): ")
basement         = get_yes_no_input("Basement? (yes/no): ")
hotwaterheating  = get_yes_no_input("Hot water heating? (yes/no): ")
airconditioning  = get_yes_no_input("Air conditioning? (yes/no): ")
parking          = int(get_numeric_input("Parking spaces (0–10): ", 0, 10))
prefarea         = get_yes_no_input("Preferred area? (yes/no): ")
furnishingstatus = get_furnishing_input()


# ─────────────────────────────────────────────────────────
# Build input DataFrame and predict
# ─────────────────────────────────────────────────────────
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
price_in_lakh   = predicted_price / 100_000

print("\n" + "=" * 50)
print("   Predicted House Price")
print("=" * 50)
print(f"   Rs {price_in_lakh:.2f} Lakh  (Rs {predicted_price:,.0f})")
print("=" * 50)
print("\nNote: This is an estimate based on the training dataset.")
print("      It should not be used as a professional property valuation.\n")