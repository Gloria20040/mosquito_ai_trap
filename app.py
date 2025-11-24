import os
import tempfile
import base64
import threading
import gc

# Force TensorFlow to skip XLA for faster startup
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import soundfile as sf
import librosa
from tensorflow.keras.models import load_model
from huggingface_hub import hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from moviepy.audio.io.AudioFileClip import AudioFileClip

# ----------------- FastAPI App -----------------
app = FastAPI(title="AI Mosquito Trap - MicroACDNet")

# ----------------- CORS -----------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Static Files -----------------
if os.path.isdir("static"):
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
ALLOWED_EXTENSIONS = {"wav", "mp3", "ogg", "webm", "m4a"}
MAX_UPLOAD_BYTES = 8 * 1024 * 1024  # 8 MB limit per upload
model = None
model_lock = threading.Lock()

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

def convert_webm_to_wav(webm_path: str, wav_path: str):
    try:
        audio = AudioFileClip(webm_path)
        audio.write_audiofile(wav_path, fps=8000, codec='pcm_s16le', logger=None)
        audio.close()
        return wav_path
    except Exception as e:
        raise RuntimeError(f"Conversion error: {e}")


def _load_audio_with_soundfile(path: str, target_sr: int = 8000, duration_s: float = 1.0):
    try:
        data, sr = sf.read(path, dtype='float32')
    except Exception:
        try:
            data, sr = librosa.load(path, sr=None, mono=False)
        except Exception as e:
            raise RuntimeError(f"Could not load audio file: {e}")

    if isinstance(data, np.ndarray) and data.ndim > 1:
        data = np.mean(data, axis=1)

    if sr != target_sr:
        data = librosa.resample(data.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    target_len = int(target_sr * duration_s)
    if data.shape[0] < target_len:
        data = np.pad(data, (0, target_len - data.shape[0]))
    else:
        data = data[:target_len]

    return data, sr

def preprocess_audio_file(file_path: str):
    y, sr = _load_audio_with_soundfile(file_path, target_sr=8000, duration_s=1.0)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=512, hop_length=160, n_mels=40)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_norm = ((mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)).T

    if mel_norm.shape[0] < TARGET_SHAPE[0]:
        mel_norm = np.pad(mel_norm, ((0, TARGET_SHAPE[0] - mel_norm.shape[0]), (0, 0)), "constant")
    elif mel_norm.shape[0] > TARGET_SHAPE[0]:
        mel_norm = mel_norm[:TARGET_SHAPE[0], :]

    return mel_norm[np.newaxis, ..., np.newaxis]

def preprocess_audio_from_base64(b64_audio: str, original_ext: str = ".webm"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=original_ext) as tmp:
        tmp.write(base64.b64decode(b64_audio))
        tmp_path = tmp.name

    try:
        if tmp_path.lower().endswith('.webm'):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as wav_tmp:
                wav_path = wav_tmp.name
            try:
                convert_webm_to_wav(tmp_path, wav_path)
                tensor = preprocess_audio_file(wav_path)
            finally:
                for p in (tmp_path, wav_path):
                    try: os.remove(p)
                    except Exception: pass
        else:
            try:
                tensor = preprocess_audio_file(tmp_path)
            finally:
                try: os.remove(tmp_path)
                except Exception: pass
    except Exception as e:
        try: os.remove(tmp_path)
        except Exception: pass
        raise
    return tensor

def run_prediction(input_tensor):
    global model
    if model is None:
        raise RuntimeError("Model not loaded")

    with model_lock:
        preds = model.predict(input_tensor)

    labels = ["non_vector", "malaria_vector"]
    idx = int(np.argmax(preds[0]))
    result = {
        "prediction": {
            "species": labels[idx],
            "probability": float(preds[0][idx]),
            "all_scores": {l: float(p) for l, p in zip(labels, preds[0])}
        }
    }

    del preds
    gc.collect()
    return result

# ----------------- API Endpoints -----------------
@app.post("/predict/file")
async def predict_file(audio_file: UploadFile = File(...)):
    if model is None:
        return JSONResponse(status_code=503, content={"error": "Model not loaded."})
    try:
        ext = (audio_file.filename or "").split(".")[-1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            return JSONResponse(status_code=400, content={"error": f"Unsupported file type: {ext}"})

        content = await audio_file.read()
        if not content:
            return JSONResponse(status_code=400, content={"error": "Uploaded file is empty."})

        if len(content) > MAX_UPLOAD_BYTES:
            return JSONResponse(status_code=413, content={"error": "File too large"})

        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            if tmp_path.lower().endswith('.webm'):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as wav_tmp:
                    wav_path = wav_tmp.name
                try:
                    convert_webm_to_wav(tmp_path, wav_path)
                    input_tensor = preprocess_audio_file(wav_path)
                finally:
                    for p in (tmp_path, wav_path):
                        try: os.remove(p)
                        except Exception: pass
            else:
                input_tensor = preprocess_audio_file(tmp_path)
                try: os.remove(tmp_path)
                except Exception: pass

            return run_prediction(input_tensor)
        except Exception as e:
            try: os.remove(tmp_path)
            except Exception: pass
            return JSONResponse(status_code=500, content={"error": str(e)})
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

        original_ext = data.get("ext", ".webm")
        if len(data["audio_base64"]) * 3 / 4 > MAX_UPLOAD_BYTES:
            return JSONResponse(status_code=413, content={"error": "File too large"})

        input_tensor = preprocess_audio_from_base64(data["audio_base64"], original_ext=original_ext)
        return run_prediction(input_tensor)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

