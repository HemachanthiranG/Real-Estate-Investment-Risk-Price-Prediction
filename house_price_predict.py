import joblib
import numpy as np
import pandas as pd

# Load trained model and selected feature names
model = joblib.load("best_house_price_model.pkl")
selected_features = joblib.load("selected_features.pkl")  # Load selected features

# Function to Predict House Price
def predict_house_price(**kwargs):
    # Convert input to DataFrame with correct feature names
    input_df = pd.DataFrame([kwargs])
    
    # Ensure only selected features are used
    input_df = input_df[selected_features]
    
    # Predict price
    predicted_price = model.predict(input_df)[0]
    return round(predicted_price, 2)

# Example User Input
user_input = {
    "Overall Qual": 8,
    "Overall Cond": 6,
    "Lot Area": 9000,
    "Gr Liv Area": 1800,
    "Full Bath": 2,
    "Half Bath": 3,
    "Garage Cars": 1
}

predicted_price = predict_house_price(**user_input)
print(f"🏠 Predicted House Price: ${predicted_price}")
