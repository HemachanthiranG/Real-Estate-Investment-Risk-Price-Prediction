import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load cleaned data
df = pd.read_csv("cleaned_real_estate_data.csv")

# 1. Create New Features
df["House Age"] = df["Yr Sold"] - df["Year Built"]
df["Remodel Age"] = df["Yr Sold"] - df["Year Remod/Add"]
df["Price per SqFt"] = df["SalePrice"] / df["Gr Liv Area"]

# 2. Encode Categorical Variables
categorical_cols = df.select_dtypes(include=["object"]).columns
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Store the encoder if needed later

# 3. Save Processed Data
df.to_csv("processed_data.csv", index=False)
print("Feature Engineering Completed! Data saved as 'processed_data.csv'.")
