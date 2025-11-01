---
title: AI Smart Mosquito Trap
emoji: ⚡
colorFrom: yellow
colorTo: red
sdk: static
pinned: false
license: mit
short_description: An AI mosquito trap that neutralizes malaria vectors
---
title: AI-Smart-Mosquito-Trap
sdk: static
app_file: app.py

AI Smart Mosquito Trap Assessor

This is a school project built using FastAPI and TensorFlow/Keras to classify mosquito sounds as either non-vector or malaria vector, based on a captured audio sample.

The application uses the app.py file to serve the prediction API and the custom index.html file to provide the frontend user interface.

How to Use

Record: Click "Start Recording" to capture a mosquito buzz (ideally 1 second long).

Stop: Click "Stop Recording."

play: "listen to your audio recording"

Predict: Click "Predict Recorded Audio" to send the sample to the AI model.

Upload: Alternatively, use the "Upload Audio File" section to analyze an existing .wav, .mp3, or .webm file.

The model will classify the sound and display a confidence score.
