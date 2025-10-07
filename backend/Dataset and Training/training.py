import cv2
import mediapipe as mp
import os
import csv
import re

# Setup MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

# Define image folder path
IMAGE_FOLDER = "sign_dataset/"  # Change to your folder path
CSV_FILE = "sign_dataset/landmarks_from_images.csv"

# Create CSV file
with open(CSV_FILE, "w", newline="") as file:
    csv_writer = csv.writer(file)
    
    # Write header: Label + 63 feature columns (X, Y, Z for 21 landmarks)
    csv_writer.writerow(["label"] + [f"x{i}" for i in range(21)] + 
                         [f"y{i}" for i in range(21)] + [f"z{i}" for i in range(21)])

    # Process all images in the folder
    for img_name in os.listdir(IMAGE_FOLDER):
        if img_name.endswith(".jpg") or img_name.endswith(".png"):
            # Extract label from filename (e.g., "A_718.jpg" → "A")
            label_match = re.match(r"([A-Z])_\d+", img_name)
            if label_match:
                label = label_match.group(1)
            else:
                continue  # Skip if label extraction fails

            # Load image
            img_path = os.path.join(IMAGE_FOLDER, img_name)
            image = cv2.imread(img_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process with MediaPipe
            results = hands.process(image_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Extract X, Y, Z coordinates
                    landmarks = []
                    for lm in hand_landmarks.landmark:
                        landmarks.extend([lm.x, lm.y, lm.z])
                    
                    # Save to CSV
                    csv_writer.writerow([label] + landmarks)

print(f"✅ Dataset saved to {CSV_FILE}")

