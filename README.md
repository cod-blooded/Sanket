# ISL Hybrid Translator (End-to-End)

This repository now includes a complete full-stack implementation:

- `server/` - FastAPI inference service using your trained static and motion models
- `client/` - Next.js webcam UI that streams frames to backend and builds sentence output

## Quick Start

## 1) Backend

Recommended Python version: 3.10 or 3.11.

```bash
cd server
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

## 2) Frontend

```bash
cd client
npm install
cp .env.local.example .env.local
npm run dev
```

Open http://localhost:3000.

## Model Files Used

- `Static-Model-Sentence/model.p`
- `Motion-LSTM-Model/motion_model.h5`
- Optional: `Motion-LSTM-Model/motion_data.pickle` (if absent, backend uses notebook action fallback labels)

## API Endpoints

- `GET /health`
- `GET /meta`
- `POST /predict`
- `POST /control`
