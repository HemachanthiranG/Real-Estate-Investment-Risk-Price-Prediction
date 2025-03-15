import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load processed data
df = pd.read_csv("processed_data.csv")

# Select multiple features for risk analysis
risk_features = ["SalePrice", "Price per SqFt", "Overall Qual", "Overall Cond", "Lot Area", "Gr Liv Area"]
df_risk = df[risk_features].copy()

# Normalize features
scaler = StandardScaler()
df_risk_scaled = scaler.fit_transform(df_risk)

# Train K-Means Clustering with multiple features
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Risk Level"] = kmeans.fit_predict(df_risk_scaled)

# Map clusters to readable risk levels
risk_mapping = {df.groupby("Risk Level")["SalePrice"].mean().idxmin(): "High Risk",
                df.groupby("Risk Level")["SalePrice"].mean().idxmax(): "Low Risk"}
risk_mapping = {key: risk_mapping.get(key, "Medium Risk") for key in range(3)}
df["Risk Level"] = df["Risk Level"].map(risk_mapping)

# Save model and scaler
joblib.dump(kmeans, "kmeans_risk_model.pkl")
joblib.dump(scaler, "scaler.pkl")

# Save processed data with risk levels
df.to_csv("risk_analysis.csv", index=False)

# Function to Predict Risk Level for New Properties
def predict_risk(sale_price, price_per_sqft, overall_qual, overall_cond, lot_area, gr_liv_area):
    kmeans_model = joblib.load("kmeans_risk_model.pkl")
    scaler = joblib.load("scaler.pkl")
    
    # Prepare input with multiple features
    input_data = np.array([[sale_price, price_per_sqft, overall_qual, overall_cond, lot_area, gr_liv_area]])
    input_scaled = scaler.transform(input_data)
    
    # Predict cluster
    cluster = kmeans_model.predict(input_scaled)[0]
    risk_label = risk_mapping.get(cluster, "Medium Risk")
    
    return risk_label

# Example User Input Prediction
user_input = {
    "sale_price": 200000,
    "price_per_sqft": 120,
    "overall_qual": 6,
    "overall_cond": 5,
    "lot_area": 8000,
    "gr_liv_area": 1600
}

predicted_risk = predict_risk(**user_input)
print(f"Predicted Risk Level: {predicted_risk}")

# Visualize Clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df["Gr Liv Area"], y=df["SalePrice"], hue=df["Risk Level"], palette="coolwarm")
plt.title("High-Risk Property Clustering")
plt.xlabel("Above Ground Living Area (sq ft)")
plt.ylabel("Sale Price")
plt.show()

print("Risk Analysis Completed! Model saved as 'kmeans_risk_model.pkl'.")
