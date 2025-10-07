import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Load CSV
csv_file = "landmarks.csv"  # Adjust filename if needed
df = pd.read_csv(csv_file)

# Extract labels and pixel data
labels = df['label'].values
images = df.iloc[:, 1:].values  # All columns except 'label'

# Reshape into 28x28 images
images = images.reshape(-1, 28, 28, 1).astype("float32") / 255.0  # Normalize

# Display an example image
plt.imshow(images[0].reshape(28, 28), cmap="gray")
plt.title(f"Label: {labels[0]}")
plt.show()
