
---

```markdown
  🧠 Indian Sign Language Recognition – MinProject

This project is a   Sign Language Detection System   developed by   CSE B.Tech students at Mangalam College of Engineering  .  
It uses   MediaPipe  ,   TensorFlow  , and   OpenCV   to recognize hand gestures representing numbers and alphabets in Indian Sign Language.  
The detected sign is then sent to an   Express.js backend   in real time.

---

📌 Table of Contents
- Overview
- Features
- Project Structure
- Tech Stack
- Installation
- Usage
- Dataset
- Express.js API
- Demo
- Contributors

---

  📖 Overview

This system detects hand gestures from a live webcam feed using   MediaPipe Hands  , extracts landmark points, and classifies the gesture using a trained   TensorFlow model  .  
The predicted character is displayed on the screen and sent to an   Express.js server  , which stores and exposes the latest prediction through a simple REST API.

---

  ✨ Features

- ✅ Real-time hand gesture detection using webcam  
- 🧠 TensorFlow model for classifying numbers (1–9) and alphabets (A–Z)  
- 🔄 Normalization and preprocessing of hand landmarks  
- 🌐 Express.js backend to store and serve the latest prediction  
- 🖼️ Automatic image saving on every prediction change  
- 🚀 Fast and lightweight implementation

---

  📂 Project Structure



sign-language-project/
├── sign_dataset/
│   └── landmarks_from_images.csv     Landmark dataset used for training
├── landmarks_from_images.h5          Trained TensorFlow model
├── server/
│   └── index.js                      Express.js backend server
├── main.py                           Main Python script (detection & prediction)
├── o.jpg                             Saved image of last prediction
└── README.md                         Project documentation
```
````

---

  🧰 Tech Stack

  Frontend / ML  :
- Python
- OpenCV
- MediaPipe
- TensorFlow / Keras
- Pandas, NumPy

  Backend  :
- Node.js
- Express.js
- Body-parser
- CORS

---

  🛠️ Installation

   1. Clone the Repository

```bash
https://github.com/rahulrajancc/CSE_MinProject.git
cd CSE_MinProject
````

   2. Install Python Dependencies

```bash
pip install -r requirements.txt
or
pip install opencv-python mediapipe tensorflow pandas numpy requests
```
```
Optional (if you're using virtual environment)
python3 -m venv venv
source venv/bin/activate   # Linux / macOS
# or
venv\Scripts\activate      # Windows

pip install -r requirements.txt

```
   3. Install Node.js Dependencies

```bash
cd server
npm install express cors body-parser
```

---

  ▶️ Usage

   1. Start the Express.js Server

```bash
cd server
node index.js
```

Server will run on   [http://localhost:7000](http://localhost:7000)  

---

   2. Run the Python Sign Detection

```bash
python main.py
```

  A webcam window will open.
  Show hand gestures (numbers 1–9 or alphabets A–Z).
  The predicted sign will display on the screen and be sent to the backend.
  Press   Esc   to exit.

---

  📊 Dataset

The model was trained on a CSV file containing   hand landmarks   extracted from images:

```python
import pandas as pd
df = pd.read_csv("sign_dataset/landmarks_from_images.csv")
print(df.shape)    Expected (N, 64): 1 label + 63 features
print(df.head())
```

---

  🌐 Express.js API

   `POST /predict`

Saves the latest prediction.

```json
{
  "prediction": "A",
  "image": "o.jpg"
}
```

   `GET /latest`

Fetches the most recent prediction.

  Response:  

```json
{
  "latestPrediction": "A"
}
```

---

  🧪 Demo

  Show hand gestures in front of the webcam.
  The character will appear on the top-left of the screen.
  The latest prediction can be viewed by opening:

```
http://localhost:7000/latest
```

---

  👨‍💻 Contributors

  B.Tech CSE Students  
  Mangalam College of Engineering  

  Rahul Rajan
  Shoan Kurien Johnson
  Sohil Suman
  Shijin Varghese

---

  📜 License

This project is for   educational purposes only   as part of the   MinProject   submission.



