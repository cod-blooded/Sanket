# ISL Hybrid Inference API

This backend serves your existing models:

- `../Static-Model-Sentence/model.p` (static hand sign classifier)
- `../Motion-LSTM-Model/motion_model.h5` (motion sequence classifier)

## 1) Create environment and install deps

```bash
cd server
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Recommended Python version: 3.10 or 3.11 for best compatibility with MediaPipe/TensorFlow wheels.
The static model was serialized with scikit-learn 1.5.2, so requirements pin that exact version.

If your environment was previously installed with different package versions, reset once:

```bash
deactivate 2>/dev/null || true
rm -rf .venv
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Run API

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

## 3) Endpoints

- `GET /health` - service and model metadata
- `GET /meta` - classes/actions and runtime settings
- `POST /predict` - live frame inference
- `POST /control` - sentence editing actions (`clear`, `backspace`, `delete_word`, `reset_session`)

## 4) Request example for `/predict`

```json
{
    "session_id": "my-session-id",
    "mode": "static",
    "image_data": "data:image/jpeg;base64,..."
}
```

## Notes

- If `motion_data.pickle` is missing, the API falls back to actions used in your notebooks: `HELLO`, `YES`, `NO`, `THANKS`, `OK`.
- The API keeps per-session state in memory (buffer, sequence, sentence) to mimic your notebook behavior.
- TensorFlow CUDA warnings are expected on CPU-only machines and are not fatal.
