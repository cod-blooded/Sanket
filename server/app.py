from __future__ import annotations

import base64
import os
import pickle
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import load_model

Mode = Literal["static", "motion"]


class LegacyCompatibleLSTM(LSTM):
    """Keras 3 compatibility shim for older H5 models.

    Some legacy TensorFlow/Keras LSTM configs include `time_major`,
    which is no longer accepted by newer Keras LSTM constructors.
    """

    def __init__(self, *args, **kwargs):
        kwargs.pop("time_major", None)
        super().__init__(*args, **kwargs)


def load_legacy_motion_model(model_path: Path):
    return load_model(
        model_path,
        compile=False,
        custom_objects={"LSTM": LegacyCompatibleLSTM},
    )


class PredictRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    mode: Mode = "static"
    image_data: str = Field(min_length=16)


class PredictResponse(BaseModel):
    session_id: str
    mode: Mode
    hand_detected: bool
    current_prediction: str | None
    confidence: float | None
    landmarks: list[list[float]]
    sentence: str
    added_to_sentence: bool
    buffer_progress: int
    sequence_progress: int


class ControlRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=128)
    action: Literal["clear", "backspace", "delete_word", "reset_session"]


class SessionResponse(BaseModel):
    session_id: str
    sentence: str


@dataclass
class SessionState:
    buffer: deque[str] = field(default_factory=lambda: deque(maxlen=10))
    sequence: deque[list[float]] = field(default_factory=lambda: deque(maxlen=30))
    motion_probs: deque[list[float]] = field(default_factory=lambda: deque(maxlen=5))
    sentence: str = ""
    prev_char: str = ""
    last_added_time: float = 0.0
    prev_motion_features: list[float] | None = None

    def clear_all(self) -> None:
        self.buffer.clear()
        self.sequence.clear()
        self.motion_probs.clear()
        self.sentence = ""
        self.prev_char = ""
        self.last_added_time = 0.0
        self.prev_motion_features = None


class HybridPredictor:
    def __init__(self) -> None:
        root = Path(__file__).resolve().parents[1]
        static_path = root / "Static-Model-Sentence" / "model.p"
        motion_path = root / "Motion-LSTM-Model" / "motion_model.h5"
        motion_data_path = root / "Motion-LSTM-Model" / "motion_data.pickle"

        if not static_path.exists():
            raise FileNotFoundError(f"Missing static model file: {static_path}")
        if not motion_path.exists():
            raise FileNotFoundError(f"Missing motion model file: {motion_path}")

        static_blob = pickle.load(static_path.open("rb"))
        self.static_model = static_blob["model"] if isinstance(static_blob, dict) else static_blob
        self.motion_model = load_legacy_motion_model(motion_path)

        self.actions, self.motion_actions_source = self._load_actions(motion_data_path)
        self.actions = self._align_actions_with_model(self.actions)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            min_detection_confidence=0.5,
            max_num_hands=1,
        )

        self._hands_lock = threading.Lock()
        self._sessions_lock = threading.Lock()
        self._sessions: dict[str, SessionState] = {}

        self._static_feature_size = int(getattr(self.static_model, "n_features_in_", 42))
        self._motion_feature_size = int(self.motion_model.input_shape[-1])
        self._warm_up_models()

    def _load_actions(self, motion_data_path: Path) -> tuple[list[str], str]:
        env_actions = os.getenv("MOTION_ACTIONS", "").strip()
        if env_actions:
            parsed = [action.strip() for action in env_actions.split(",") if action.strip()]
            if parsed:
                return parsed, "env"

        if motion_data_path.exists():
            data = pickle.load(motion_data_path.open("rb"))
            if isinstance(data, dict) and "actions" in data:
                return [str(a) for a in data["actions"]], "motion_data"

        # Fallback to actions used in your data collection notebook.
        return ["HELLO", "YES", "NO", "THANKS", "OK"], "fallback"

    def _align_actions_with_model(self, actions: list[str]) -> list[str]:
        output_shape = self.motion_model.output_shape
        output_units = int(output_shape[-1]) if isinstance(output_shape, tuple) else int(output_shape[0][-1])

        if len(actions) == output_units:
            return actions
        if len(actions) > output_units:
            return actions[:output_units]

        filled = list(actions)
        for index in range(len(actions), output_units):
            filled.append(f"ACTION_{index + 1}")
        return filled

    @staticmethod
    def _decode_image(image_data: str) -> np.ndarray:
        payload = image_data.split(",", 1)[1] if "," in image_data else image_data

        try:
            raw = base64.b64decode(payload)
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Invalid base64 image payload") from exc

        array = np.frombuffer(raw, dtype=np.uint8)
        bgr = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if bgr is None:
            raise HTTPException(status_code=400, detail="Could not decode image payload")

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # Notebook capture/inference uses a horizontally flipped frame.
        return cv2.flip(rgb, 1)

    @staticmethod
    def _normalize_feature_length(features: list[float], target_size: int) -> list[float]:
        if len(features) == target_size:
            return features
        if len(features) > target_size:
            return features[:target_size]

        return features + ([0.0] * (target_size - len(features)))

    @staticmethod
    def _landmarks_xy(hand_landmarks) -> list[list[float]]:
        return [[float(lm.x), float(lm.y)] for lm in hand_landmarks.landmark]

    @staticmethod
    def _extract_hand_features(hand_landmarks) -> list[float]:
        x_values = []
        y_values = []

        for lm in hand_landmarks.landmark:
            x_values.append(lm.x)
            y_values.append(lm.y)

        data_aux: list[float] = []
        min_x = min(x_values)
        min_y = min(y_values)

        for lm in hand_landmarks.landmark:
            data_aux.append(lm.x - min_x)
            data_aux.append(lm.y - min_y)

        return data_aux

    def _predict_features(self, image_rgb: np.ndarray) -> tuple[list[float] | None, list[list[float]]]:
        with self._hands_lock:
            results = self.hands.process(image_rgb)

        if not results.multi_hand_landmarks:
            return None, []

        hand = results.multi_hand_landmarks[0]
        return self._extract_hand_features(hand), self._landmarks_xy(hand)

    def _warm_up_models(self) -> None:
        try:
            static_stub = np.zeros((1, self._static_feature_size), dtype=np.float32)
            self.static_model.predict(static_stub)
        except Exception:
            pass

        try:
            motion_stub = np.zeros((1, 30, self._motion_feature_size), dtype=np.float32)
            self.motion_model.predict(motion_stub, verbose=0)
        except Exception:
            pass

    def _get_session(self, session_id: str) -> SessionState:
        with self._sessions_lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionState()
            return self._sessions[session_id]

    def predict(self, request: PredictRequest) -> PredictResponse:
        session = self._get_session(request.session_id)
        image_rgb = self._decode_image(request.image_data)
        features, landmarks = self._predict_features(image_rgb)

        if features is None:
            if request.mode == "motion":
                session.buffer.clear()
                if session.prev_motion_features is not None:
                    session.sequence.append(session.prev_motion_features)
                else:
                    session.sequence.append([0.0] * self._motion_feature_size)

            return PredictResponse(
                session_id=request.session_id,
                mode=request.mode,
                hand_detected=False,
                current_prediction=None,
                confidence=None,
                landmarks=[],
                sentence=session.sentence,
                added_to_sentence=False,
                buffer_progress=len(session.buffer),
                sequence_progress=len(session.sequence),
            )

        current_time = time.time()
        current_prediction: str | None = None
        confidence: float | None = None
        added_to_sentence = False

        if request.mode == "static":
            session.sequence.clear()
            session.motion_probs.clear()
            session.prev_motion_features = None
            features = self._normalize_feature_length(features, self._static_feature_size)
            raw_prediction = self.static_model.predict([features])[0]
            current_prediction = str(raw_prediction)
            session.buffer.append(current_prediction)

            if hasattr(self.static_model, "predict_proba"):
                probabilities = self.static_model.predict_proba([features])[0]
                confidence = float(np.max(probabilities))

            if len(session.buffer) == session.buffer.maxlen:
                stable_char = Counter(session.buffer).most_common(1)[0][0]
                current_prediction = stable_char

                if stable_char != session.prev_char and (current_time - session.last_added_time) > 1.0:
                    if stable_char == "space":
                        session.sentence += " "
                    else:
                        session.sentence += stable_char

                    session.prev_char = stable_char
                    session.last_added_time = current_time
                    added_to_sentence = True

        else:
            session.buffer.clear()
            features = self._normalize_feature_length(features, self._motion_feature_size)
            session.prev_motion_features = features
            session.sequence.append(features)

            if len(session.sequence) == session.sequence.maxlen:
                sequence_np = np.array(session.sequence, dtype=np.float32)
                prediction_vector = self.motion_model.predict(
                    np.expand_dims(sequence_np, axis=0),
                    verbose=0,
                )[0]
                session.motion_probs.append(prediction_vector.tolist())
                smooth_vector = np.mean(np.array(session.motion_probs, dtype=np.float32), axis=0)

                action_idx = int(np.argmax(smooth_vector))
                current_prediction = self.actions[action_idx]
                confidence = float(smooth_vector[action_idx])

                if confidence >= 0.55 and (current_time - session.last_added_time) > 1.0:
                    if session.sentence and not session.sentence.endswith(" "):
                        session.sentence += " "
                    session.sentence += f"{current_prediction} "
                    session.last_added_time = current_time
                    session.prev_char = ""
                    added_to_sentence = True

        if len(session.sentence) > 120:
            session.sentence = session.sentence[-120:]

        return PredictResponse(
            session_id=request.session_id,
            mode=request.mode,
            hand_detected=True,
            current_prediction=current_prediction,
            confidence=confidence,
            landmarks=landmarks,
            sentence=session.sentence,
            added_to_sentence=added_to_sentence,
            buffer_progress=len(session.buffer),
            sequence_progress=len(session.sequence),
        )

    def control(self, request: ControlRequest) -> SessionResponse:
        session = self._get_session(request.session_id)

        if request.action in {"clear", "reset_session"}:
            session.clear_all()
        elif request.action == "backspace":
            session.sentence = session.sentence[:-1]
        elif request.action == "delete_word":
            session.sentence = session.sentence.rstrip()
            session.sentence = " ".join(session.sentence.split(" ")[:-1])

        return SessionResponse(session_id=request.session_id, sentence=session.sentence)

    def metadata(self) -> dict[str, object]:
        static_classes = []
        if hasattr(self.static_model, "classes_"):
            static_classes = [str(c) for c in self.static_model.classes_]

        return {
            "modes": ["static", "motion"],
            "static_classes": static_classes,
            "motion_actions": self.actions,
            "motion_actions_source": self.motion_actions_source,
            "buffer_size": 10,
            "sequence_length": 30,
            "cooldown_seconds": 1.0,
        }


app = FastAPI(title="ISL Hybrid Inference API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

predictor = HybridPredictor()


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "meta": predictor.metadata()}


@app.get("/meta")
def meta() -> dict[str, object]:
    return predictor.metadata()


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    return predictor.predict(request)


@app.post("/control", response_model=SessionResponse)
def control(request: ControlRequest) -> SessionResponse:
    return predictor.control(request)
