const express = require("express");
const cors = require("cors");
const bodyParser = require("body-parser");

const app = express();
app.use(cors());
app.use(bodyParser.json());

let latestPrediction = ""; // Store the latest prediction

// POST API to receive predictions
app.post("/predict", (req, res) => {
    const { prediction } = req.body;
    console.log("Received Prediction:", prediction);
    latestPrediction = prediction; // Store latest prediction
    res.json({ message: "Prediction received", prediction });
});

// GET API to fetch the latest prediction
app.get("/latest", (req, res) => {
    res.json({ latestPrediction });
});

// Home Route
app.get("/", (req, res) => {
    res.send("<h1>Welcome to the Sign Language API</h1>");
});

// Start Server on port 7000
app.listen(7000, () => {
    console.log("Server running on http://localhost:7000");
});
