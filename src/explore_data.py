
import pandas as pd

# Load dataset
df = pd.read_excel(
    "data/data.xls",
    header=1
)

# 1. Dataset size
print("Dataset Shape:")
print(df.shape)

# 2. Column names
print("\nColumns:")
print(df.columns.tolist())

# 3. First 5 records
print("\nFirst 5 Records:")
print(df.head())

# 4. Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# 5. Duplicate records
print("\nDuplicate Records:")
print(df.duplicated().sum())

# 6. Statistical summary
print("\nStatistical Summary:")
print(df.describe())