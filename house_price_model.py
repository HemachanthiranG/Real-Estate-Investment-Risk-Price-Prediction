import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

df = pd.read_csv("processed_data.csv")

features = ["Overall Qual", "Overall Cond", "Lot Area", "Gr Liv Area", "Full Bath", "Half Bath", "Garage Cars"]
target = "SalePrice"

X = df[features]
y = df[target]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


selector = SelectKBest(score_func=f_regression, k=5)  # Select top 5 features
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)
selected_features = X_train.columns[selector.get_support()]

# Save the selected features
joblib.dump(selected_features, "selected_features.pkl")
print(f"🔹 Selected Features: {selected_features.tolist()}")

# Train Random Forest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_selected, y_train)

# Save the trained model
joblib.dump(model, "best_house_price_model.pkl")
print("✅ Model and selected features saved successfully!")
