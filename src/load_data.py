
import pandas as pd
import os

file_path = "data/data.xls"

# Check whether dataset exists
if os.path.exists(file_path):
    print("Dataset found!")

    # Load dataset
    df = pd.read_excel(file_path, header=1)

    # Display first 5 rows
    print(df.head())

    # Dataset information
    print("\nDataset Shape:")
    print(df.shape)

    print("\nColumn Names:")
    print(df.columns.tolist())

    print("\nDataset Information:")
    df.info()

else:
    print("Dataset not found. Check your file location.")
