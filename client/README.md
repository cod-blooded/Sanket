# ISL Client App

This Next.js app provides a live webcam UI for your ISL hybrid classifier backend.

## 1) Install dependencies

```bash
cd client
npm install
```

## 2) Configure backend URL

Create `.env.local` from `.env.local.example`:

```bash
cp .env.local.example .env.local
```

Default value:

```env
NEXT_PUBLIC_INFERENCE_API_URL=http://127.0.0.1:8000
```

## 3) Run frontend

```bash
npm run dev
```

## 4) Start backend (required)

In another terminal:

```bash
cd ../server
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

## Features

- Live webcam feed
- Static and motion inference modes
- Real-time prediction and confidence
- Sentence composition with controls (clear, backspace, delete word)
- API status indicator
