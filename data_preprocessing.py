import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Load dataset
data = pd.read_csv('real_estate_data.csv')

# Handle missing values
# Fill missing LotFrontage with median value per Neighborhood
data['Lot Frontage'] = data.groupby('Neighborhood')['Lot Frontage'].transform(lambda x: x.fillna(x.median()))

# Fill categorical missing values with 'None'
categorical_cols = ['Alley', 'Mas Vnr Type', 'Fireplace Qu', 'Garage Type', 'Garage Finish', 'Garage Qual', 'Garage Cond', 'Pool QC', 'Fence', 'Misc Feature']
for col in categorical_cols:
    data[col].fillna('None', inplace=True)

# Fill numerical missing values with 0
numerical_cols = ['Mas Vnr Area', 'BsmtFin SF 1', 'BsmtFin SF 2', 'Bsmt Unf SF', 'Total Bsmt SF', 'Garage Yr Blt', 'Garage Cars', 'Garage Area']
for col in numerical_cols:
    data[col].fillna(0, inplace=True)
# Detect Outliers using IQR method
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]

# Columns to check for outliers
num_cols = ["SalePrice", "Lot Area", "Gr Liv Area"]

# Plot boxplots for outlier detection
plt.figure(figsize=(12, 6))
for i, col in enumerate(num_cols):
    plt.subplot(1, 3, i+1)
    sns.boxplot(y=data[col])
    plt.title(f"Boxplot of {col}")

plt.tight_layout()
plt.show()

# Print detected outliers
for col in num_cols:
    outliers = detect_outliers_iqr(data, col)
    print(f"🔴 Outliers in {col}: {len(outliers)} found")

# Handling Outliers
for col in num_cols:
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Option 1: Remove extreme outliers
    data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]

    # Option 2: Cap extreme values instead of removing
    # data[col] = np.clip(data[col], lower_bound, upper_bound)

# Save cleaned data
data.to_csv('cleaned_real_estate_data.csv', index=False)
print("Data preprocessing completed and saved.")
