import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# 1️⃣ Load the dataset
csv_file = "sign_mnist_train.csv"  # Adjust path if needed
df = pd.read_csv(csv_file)

# 2️⃣ Extract labels and pixel values
labels = df['label'].values  # Define labels
images = df.iloc[:, 1:].values  # Exclude the 'label' column

# 3️⃣ Reshape & normalize images
images = images.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# 4️⃣ Convert labels to one-hot encoding
y_train = to_categorical(labels, num_classes=26)

# 5️⃣ Split into training & validation sets
X_train, X_val, y_train, y_val = train_test_split(images, y_train, test_size=0.2, random_state=42)

# 6️⃣ Define CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(26, activation="softmax")  # 26 classes (A-Z)
])

# 7️⃣ Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# 8️⃣ Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# 9️⃣ Save the trained model
model.save("sign_language_model.h5")

print("✅ Model training complete!")
