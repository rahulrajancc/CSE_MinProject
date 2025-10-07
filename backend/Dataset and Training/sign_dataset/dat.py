import pandas as pd


# Download dataset


# Locate the CSV file (Adjust filename based on actual dataset)
csv_file =  "landmarks.csv"

# Load data into Pandas DataFrame
df = pd.read_csv(csv_file)

print("Dataset Loaded Successfully!")
print(df.head())  # Display first 5 rows
