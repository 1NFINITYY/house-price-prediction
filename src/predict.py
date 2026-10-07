import joblib
import pandas as pd
from pathlib import Path

# Get project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Load trained model
model_path = BASE_DIR / "models" / "house_price_model.pkl"

model = joblib.load(model_path)

# New house
# Get user input
area = float(input("Enter area (sq ft): "))
bedrooms = int(input("Enter bedrooms: "))
bathrooms = int(input("Enter bathrooms: "))
stories = int(input("Enter stories: "))

mainroad = input("Main road? (yes/no): ")
guestroom = input("Guestroom? (yes/no): ")
basement = input("Basement? (yes/no): ")
hotwaterheating = input("Hot water heating? (yes/no): ")
airconditioning = input("Air conditioning? (yes/no): ")

parking = int(input("Number of parking spaces: "))

prefarea = input("Preferred area? (yes/no): ")
furnishingstatus = input(
    "Furnishing status (furnished/semi-furnished/unfurnished): "
)

# Create input DataFrame
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

# Prediction
predicted_price = model.predict(new_house)[0]

print("\n🏠 Predicted House Price:")
print(f"₹{predicted_price:,.2f}")