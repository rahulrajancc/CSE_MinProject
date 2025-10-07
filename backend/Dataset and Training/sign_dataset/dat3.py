import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("sign_language_model.h5")

# Define labels (A-Z, assuming dataset uses 26 categories)
labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)

# OpenCV Video Capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame (mirror effect)
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert BGR to RGB for MediaPipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract hand region (bounding box)
            x_min, y_min = w, h
            x_max, y_max = 0, 0
            for lm in hand_landmarks.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_min, y_min = min(x_min, x), min(y_min, y)
                x_max, y_max = max(x_max, x), max(y_max, y)

            # Crop & resize to 28x28 (match model input size)
            hand_img = frame[y_min:y_max, x_min:x_max]
            if hand_img.size > 0:
                hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
                hand_img = cv2.resize(hand_img, (28, 28))
                hand_img = np.expand_dims(hand_img, axis=[0, -1])  # Reshape for CNN model
                hand_img = hand_img / 255.0  # Normalize

                # Predict sign
                prediction = model.predict(hand_img)
                sign_index = np.argmax(prediction)
                predicted_sign = labels[sign_index]

                # Display prediction on screen
                cv2.putText(frame, predicted_sign, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show the video feed
    cv2.imshow("Sign Language Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
