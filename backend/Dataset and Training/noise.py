import pandas as pd
df = pd.read_csv("sign_dataset/landmarks_from_images.csv")
print(df.shape)  # Should be (N, 64) → 1 label + 63 features
print(df.head())  # View first few rows

