import pandas as pd

# Load dataset
csv_file = "sign_mnist_train.csv"
df = pd.read_csv(csv_file)

# 1️⃣ Check the first few values in the label column
print("First 10 labels:", df['label'].head())

# 2️⃣ Check unique labels
unique_labels = set(df['label'])
print("Unique Labels:", unique_labels)

# 3️⃣ Detect incorrect labels (if numeric instead of A-Z)
if all(isinstance(label, (int, float)) for label in unique_labels):
    print("⚠️ Warning: Labels appear to be numerical! Replacing with placeholders.")
    
    # Replace with placeholder labels (A, B, C...Z in sequence)
    corrected_labels = [chr(65 + (i % 26)) for i in range(len(df))]  # A-Z repeating
    
    # Assign corrected labels
    df['label'] = corrected_labels

# 4️⃣ Save corrected dataset
corrected_csv_file = "/home/rahulrajan/Documents/sign_dataset/landmarks_corrected.csv"
df.to_csv(corrected_csv_file, index=False)

print(f"✅ Corrected dataset saved at: {corrected_csv_file}")

