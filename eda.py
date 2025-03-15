import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load preprocessed data
df = pd.read_csv("cleaned_real_estate_data.csv")

# Select only numerical columns
numerical_df = df.select_dtypes(include=['number'])

# Display summary statistics
print(numerical_df.describe())

# Correlation heatmap (only for numerical features)
plt.figure(figsize=(12, 8))
sns.heatmap(numerical_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# Distribution of SalePrice
plt.figure(figsize=(8, 5))
sns.histplot(df["SalePrice"], bins=50, kde=True)
plt.title("Sale Price Distribution")
plt.xlabel("SalePrice")
plt.ylabel("Frequency")
plt.show()

# Scatter plot: SalePrice vs. Living Area
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df["Gr Liv Area"], y=df["SalePrice"])
plt.title("Sale Price vs. Living Area")
plt.xlabel("Ground Living Area (sq ft)")
plt.ylabel("Sale Price")
plt.show()

# Box plot: SalePrice vs. Neighborhood (Ensure 'Neighborhood' is categorical)
if "Neighborhood" in df.columns:
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=df["Neighborhood"], y=df["SalePrice"])
    plt.xticks(rotation=90)
    plt.title("Sale Price by Neighborhood")
    plt.show()
