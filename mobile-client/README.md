# mobile-client

Expo React Native client for the ISL inference backend. It mirrors the web client flow:

- front-camera frame capture
- `/health`, `/predict`, and `/control` API calls
- static and motion modes
- live prediction, confidence, hand status, progress, sentence controls, and landmark overlay

## Setup

Install dependencies:

```sh
npm install
```

Create a local env file:

```sh
cp .env.example .env
```

Set `EXPO_PUBLIC_INFERENCE_API_URL` to the backend URL. On a real phone this must be a reachable LAN address or your ngrok forwarding URL, not `127.0.0.1`.

Example:

```sh
EXPO_PUBLIC_INFERENCE_API_URL=http://192.168.1.25:8000
```

ngrok example:

```sh
EXPO_PUBLIC_INFERENCE_API_URL=https://your-tunnel.ngrok-free.app
```

## Run

Start the backend from `../server` first:

```sh
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Then start Expo:

```sh
npm start
```

Scan the QR code with Expo Go, or run:

```sh
npm run android
npm run ios
```

If you run npm from Windows against this WSL workspace, use `cmd /c pushd` so npm gets a temporary drive letter instead of a UNC current directory:

```bat
cmd /c "pushd \\wsl.localhost\RHEL\root\projects\Sanket\mobile-client && npm start"
```

Use the same pattern for checks:

```bat
cmd /c "pushd \\wsl.localhost\RHEL\root\projects\Sanket\mobile-client && npm run typecheck"
```

## Check

```sh
npm run typecheck
```
