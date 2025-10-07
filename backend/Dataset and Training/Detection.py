import cv2
import mediapipe as mp
import copy
import itertools
from tensorflow import keras
import numpy as np
import pandas as pd
import string
import requests  # Import requests to send data to Express.js

# Load the trained model
model = keras.models.load_model("landmarks_from_images.h5")

# Initialize MediaPipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Define sign language characters
alphabet = ['1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet += list(string.ascii_uppercase)
alphabet.append("NO")  # Adding "NO" as a possible prediction

# Set image save path
image_path = "o.jpg"

# Variable to store last sent prediction
last_prediction = None  

# Function to calculate hand landmark positions
def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    for landmark in landmarks.landmark:
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

# Function to normalize landmark data
def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    base_x, base_y = temp_landmark_list[0][0], temp_landmark_list[0][1]
    for index, landmark_point in enumerate(temp_landmark_list):
        temp_landmark_list[index][0] -= base_x
        temp_landmark_list[index][1] -= base_y

    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))
    max_value = max(map(abs, temp_landmark_list)) if temp_landmark_list else 1
    temp_landmark_list = [n / max_value for n in temp_landmark_list]

    return temp_landmark_list

# Initialize webcam
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    model_complexity=0,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:
    while cap.isOpened():
        success, image = cap.read()
        image = cv2.flip(image, 1)  # Flip for a selfie-view display
        
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert image for processing
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        debug_image = copy.deepcopy(image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                pre_processed_landmark_list = pre_process_landmark(landmark_list)
                
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Convert to DataFrame
                df = pd.DataFrame(pre_processed_landmark_list).transpose()

                # Predict sign language
                predictions = model.predict(df, verbose=0)
                predicted_classes = np.argmax(predictions, axis=1)

                # Ensure the predicted index is within bounds
                if predicted_classes[0] < len(alphabet):
                    label = alphabet[predicted_classes[0]]
                else:
                    label = "UNKNOWN"

                # Only send data if prediction has changed
                if label != last_prediction:
                    last_prediction = label  # Update last prediction

                    # Save image as "o.jpg" (overwrite previous one)
                    cv2.imwrite(image_path, debug_image)
                    print(f"Image saved as {image_path}")

                    # Send prediction to Express.js server
                    url = "http://localhost:7000/predict"  # Express server endpoint
                    data = {"prediction": label, "image": image_path}

                    try:
                        response = requests.post(url, json=data)
                        print("Sent data to server:", response.json())
                    except requests.exceptions.RequestException as e:
                        print("Error sending data:", e)

                # Display prediction on screen
                cv2.putText(image, label, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

                print(label)
                print("------------------------")

        # Show output image
        cv2.imshow('Indian Sign Language Detector', image)
        if cv2.waitKey(5) & 0xFF == 27:  # Press 'Esc' to exit
            break

cap.release()
cv2.destroyAllWindows()
