import os

# ----------------- FORCE TF TO SKIP XLA (SPEEDUP STARTUP) -----------------
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import base64
import numpy as np
import librosa
from tensorflow.keras.models import load_model
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

# ----------------- FastAPI App -----------------
app = FastAPI(title="AI Mosquito Trap")

# ----------------- CORS -----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Static Files -----------------
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_homepage():
    index_path = os.path.join("static", "index.html")
    if not os.path.exists(index_path):
        return HTMLResponse("<h1>index.html not found in /static</h1>", status_code=404)
    with open(index_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)

# ----------------- Model Config -----------------
REPO_ID = "Gloria004/trap-mosquito"
FILENAME = "best_microacdnet1.keras"
HF_TOKEN_ENV_VAR = "MOSQUITO_HF_TOKEN"
MODEL_PATH = None
TARGET_SHAPE = (51, 40, 1)
ALLOWED_EXTENSIONS = {"wav", "mp3", "ogg", "webm"}
model = None

# ----------------- Startup -----------------
@app.on_event("startup")
def load_model_on_startup():
    global model, MODEL_PATH
    hf_token = os.environ.get(HF_TOKEN_ENV_VAR)
    print(f"Downloading model {FILENAME} from {REPO_ID}...")
    try:
        MODEL_PATH = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, token=hf_token)
        model = load_model(MODEL_PATH)
        print("✅ Model loaded successfully.")
    except HfHubHTTPError as e:
        raise RuntimeError("❌ Could not access Hugging Face repository.") from e
    except Exception as e:
        raise RuntimeError(f"❌ Model loading error: {e}")

# ----------------- Helpers -----------------
def preprocess_audio_file(file_path: str):
    y, sr = librosa.load(file_path, sr=8000, mono=True, duration=1.0)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512, hop_length=160, n_mels=40)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_norm = ((mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)).T

    if mel_norm.shape[0] < TARGET_SHAPE[0]:
        mel_norm = np.pad(mel_norm, ((0, TARGET_SHAPE[0]-mel_norm.shape[0]), (0,0)), "constant")
    elif mel_norm.shape[0] > TARGET_SHAPE[0]:
        mel_norm = mel_norm[:TARGET_SHAPE[0], :]
    
    return mel_norm[np.newaxis, ..., np.newaxis]

def preprocess_audio_from_base64(b64_audio: str):
    tmp_path = "temp_audio.webm"
    with open(tmp_path, "wb") as f:
        f.write(base64.b64decode(b64_audio))
    tensor = preprocess_audio_file(tmp_path)
    os.remove(tmp_path)
    return tensor

def run_prediction(input_tensor):
    preds = model.predict(input_tensor)
    labels = ["non_vector", "malaria_vector"]
    idx = np.argmax(preds[0])
    return {
        "prediction": {
            "species": labels[idx],
            "probability": float(preds[0][idx]),
            "all_scores": {l: float(p) for l, p in zip(labels, preds[0])}
        }
    }

# ----------------- API Endpoints -----------------
@app.post("/predict/file")
async def predict_file(audio_file: UploadFile = File(...)):
    if model is None:
        return JSONResponse(status_code=503, content={"error": "Model not loaded."})
    try:
        ext = audio_file.filename.split(".")[-1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            return JSONResponse(status_code=400, content={"error": f"Unsupported file type: {ext}"})
        tmp_path = f"temp_upload.{ext}"
        content = await audio_file.read()
        if not content:
            return JSONResponse(status_code=400, content={"error": "Uploaded file is empty."})
        with open(tmp_path, "wb") as f:
            f.write(content)
        input_tensor = preprocess_audio_file(tmp_path)
        os.remove(tmp_path)
        return run_prediction(input_tensor)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/predict/base64")
async def predict_base64(request: Request):
    if model is None:
        return JSONResponse(status_code=503, content={"error": "Model not loaded."})
    try:
        data = await request.json()
        if "audio_base64" not in data:
            return JSONResponse(status_code=400, content={"error": "Missing 'audio_base64' field"})
        input_tensor = preprocess_audio_from_base64(data["audio_base64"])
        return run_prediction(input_tensor)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ----------------- Run Server -----------------
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
