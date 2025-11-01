from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import base64
import numpy as np
import librosa
import os
import requests
from tensorflow.keras.models import load_model

# --- Model Configuration ---
MODEL_URL = "Gloria004/AI-Smart-Mosquito-Trap"
TARGET_SHAPE = (51, 40, 1)
ALLOWED_EXTENSIONS = {"wav", "mp3", "ogg", "webm"}

app = FastAPI()
model = None

# --- Serve static files ---
app.mount("/static", StaticFiles(directory="static"), name="static")


def download_model():
    """Download model from Hugging Face if not found locally."""
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Hugging Face...")
        r = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as f:
            f.write(r.content)
        print("✅ Model downloaded successfully!")


@app.on_event("startup")
def load_model_on_startup():
    global model
    download_model()
    model = load_model(MODEL_PATH)
    print("✅ Model loaded and ready.")


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    with open("static/index.html", "r", encoding="utf-8") as f:
        html = f.read()
    return HTMLResponse(content=html)


def preprocess_audio_file(file_path: str):
    """Load audio, compute Mel-spectrogram, normalize, pad/truncate."""
    y, sr = librosa.load(file_path, sr=8000, mono=True, duration=1.0)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512, hop_length=160, n_mels=40)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_norm = ((mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)).T

    # Pad/Truncate to target frames
    if mel_norm.shape[0] < TARGET_SHAPE[0]:
        mel_norm = np.pad(mel_norm, ((0, TARGET_SHAPE[0] - mel_norm.shape[0]), (0, 0)), "constant")
    elif mel_norm.shape[0] > TARGET_SHAPE[0]:
        mel_norm = mel_norm[:TARGET_SHAPE[0], :]
    return mel_norm[np.newaxis, ..., np.newaxis]


def preprocess_audio_from_base64(b64_audio: str):
    tmp_path = "temp_audio.webm"
    audio_data = base64.b64decode(b64_audio)
    with open(tmp_path, "wb") as f:
        f.write(audio_data)
    tensor = preprocess_audio_file(tmp_path)
    os.remove(tmp_path)
    return tensor


@app.post("/predict")
async def predict(request: Request, audio_file: UploadFile = File(None)):
    try:
        # Determine input source
        if audio_file:
            ext = audio_file.filename.split(".")[-1].lower()
            if ext not in ALLOWED_EXTENSIONS:
                return JSONResponse(status_code=400, content={"error": f"Unsupported file type: {ext}"})
            tmp_path = f"temp_upload.{ext}"
            with open(tmp_path, "wb") as f:
                f.write(await audio_file.read())
            input_tensor = preprocess_audio_file(tmp_path)
            os.remove(tmp_path)
        else:
            data = await request.json()
            if "audio_base64" not in data:
                return JSONResponse(status_code=400, content={"error": "No audio file or base64 data provided"})
            input_tensor = preprocess_audio_from_base64(data["audio_base64"])

        # Predict
        preds = model.predict(input_tensor)
        idx = np.argmax(preds[0])
        labels = ["non_vector", "malaria_vector"]
        return {"prediction": {
            "species": labels[idx],
            "probability": float(preds[0][idx]),
            "all_scores": {l: float(p) for l, p in zip(labels, preds[0])}
        }}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Prediction failed: {str(e)}"})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
