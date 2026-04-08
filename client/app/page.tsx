"use client"

import { useCallback, useEffect, useMemo, useRef, useState } from "react"

import { Button } from "@/components/ui/button"

type Mode = "static" | "motion"

type LandmarkPoint = {
    x: number
    y: number
}

type LandmarkPayloadPoint = LandmarkPoint | [number, number]

type PredictResponse = {
    session_id: string
    mode: Mode
    hand_detected: boolean
    current_prediction: string | null
    confidence: number | null
    landmarks: LandmarkPayloadPoint[]
    sentence: string
    added_to_sentence: boolean
    buffer_progress: number
    sequence_progress: number
}

const HAND_CONNECTIONS: Array<[number, number]> = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [5, 9], [9, 10], [10, 11], [11, 12],
    [9, 13], [13, 14], [14, 15], [15, 16],
    [13, 17], [17, 18], [18, 19], [19, 20],
    [0, 17],
]

const CAPTURE_WIDTH = 640
const PREDICT_INTERVAL_MS = 220
const PREDICT_TIMEOUT_MS = 8000

const API_BASE =
    process.env.NEXT_PUBLIC_INFERENCE_API_URL?.replace(/\/$/, "") ??
    "http://127.0.0.1:8000"

function normalizeLandmarkPayload(points: LandmarkPayloadPoint[]): LandmarkPoint[] {
    return points
        .map((point) => {
            if (Array.isArray(point)) {
                const [x, y] = point
                if (typeof x !== "number" || typeof y !== "number") {
                    return null
                }
                return { x, y }
            }

            if (typeof point?.x !== "number" || typeof point?.y !== "number") {
                return null
            }

            return point
        })
        .filter((point): point is LandmarkPoint => point !== null)
}

function createSessionId() {
    if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
        return crypto.randomUUID()
    }

    return `session-${Date.now()}-${Math.random().toString(16).slice(2)}`
}

export default function Page() {
    const sessionId = useMemo(() => createSessionId(), [])

    const videoRef = useRef<HTMLVideoElement | null>(null)
    const captureCanvasRef = useRef<HTMLCanvasElement | null>(null)
    const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null)
    const inFlightRef = useRef(false)

    const [mode, setMode] = useState<Mode>("static")
    const [isCameraOn, setIsCameraOn] = useState(false)
    const [isSending, setIsSending] = useState(false)
    const [serverOnline, setServerOnline] = useState<boolean | null>(null)

    const [handDetected, setHandDetected] = useState(false)
    const [currentPrediction, setCurrentPrediction] = useState("-")
    const [confidence, setConfidence] = useState<number | null>(null)
    const [sentence, setSentence] = useState("")
    const [bufferProgress, setBufferProgress] = useState(0)
    const [sequenceProgress, setSequenceProgress] = useState(0)
    const [errorMessage, setErrorMessage] = useState<string | null>(null)

    const checkHealth = useCallback(async () => {
        try {
            const response = await fetch(`${API_BASE}/health`)
            if (!response.ok) {
                throw new Error(`Health check failed with status ${response.status}`)
            }
            setServerOnline(true)
        } catch {
            setServerOnline(false)
        }
    }, [])

    useEffect(() => {
        void checkHealth()
    }, [checkHealth])

    const stopCamera = useCallback(() => {
        const stream = videoRef.current?.srcObject
        if (stream instanceof MediaStream) {
            stream.getTracks().forEach((track) => track.stop())
        }

        if (videoRef.current) {
            videoRef.current.srcObject = null
        }

        const overlay = overlayCanvasRef.current
        if (overlay) {
            const context = overlay.getContext("2d")
            context?.clearRect(0, 0, overlay.width, overlay.height)
        }

        setIsCameraOn(false)
    }, [])

    const startCamera = useCallback(async () => {
        setErrorMessage(null)

        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: "user",
                    width: { ideal: 960 },
                    height: { ideal: 720 },
                },
                audio: false,
            })

            if (!videoRef.current) {
                return
            }

            videoRef.current.srcObject = stream
            await videoRef.current.play()
            setIsCameraOn(true)
        } catch {
            setErrorMessage(
                "Camera access failed. Please allow webcam access and make sure no other app is locking it."
            )
        }
    }, [])

    useEffect(() => {
        return () => {
            stopCamera()
        }
    }, [stopCamera])

    const sendControl = useCallback(
        async (action: "clear" | "backspace" | "delete_word" | "reset_session") => {
            try {
                const response = await fetch(`${API_BASE}/control`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ session_id: sessionId, action }),
                })

                if (!response.ok) {
                    throw new Error(`Control request failed with status ${response.status}`)
                }

                const data = (await response.json()) as { sentence: string }
                setSentence(data.sentence)
            } catch {
                setErrorMessage("Failed to send control action to backend.")
            }
        },
        [sessionId]
    )

    const captureFrame = useCallback(() => {
        const video = videoRef.current
        const canvas = captureCanvasRef.current

        if (!video || !canvas || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
            return null
        }

        const width = video.videoWidth
        const height = video.videoHeight

        if (!width || !height) {
            return null
        }

        const targetWidth = Math.min(CAPTURE_WIDTH, width)
        const targetHeight = Math.round((height / width) * targetWidth)

        canvas.width = targetWidth
        canvas.height = targetHeight

        const context = canvas.getContext("2d")
        if (!context) {
            return null
        }

        context.drawImage(video, 0, 0, targetWidth, targetHeight)
        return canvas.toDataURL("image/jpeg", 0.6)
    }, [])

    const drawLandmarks = useCallback((landmarks: LandmarkPoint[]) => {
        const video = videoRef.current
        const canvas = overlayCanvasRef.current

        if (!video || !canvas || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
            return
        }

        const width = video.videoWidth
        const height = video.videoHeight
        if (!width || !height) {
            return
        }

        if (canvas.width !== width || canvas.height !== height) {
            canvas.width = width
            canvas.height = height
        }

        const context = canvas.getContext("2d")
        if (!context) {
            return
        }

        context.clearRect(0, 0, width, height)

        if (!landmarks.length) {
            return
        }

        context.lineWidth = 2
        context.strokeStyle = "rgba(34, 197, 94, 0.92)"
        context.fillStyle = "rgba(250, 204, 21, 0.95)"

        for (const [startIdx, endIdx] of HAND_CONNECTIONS) {
            const start = landmarks[startIdx]
            const end = landmarks[endIdx]

            if (!start || !end) {
                continue
            }

            context.beginPath()
            context.moveTo(start.x * width, start.y * height)
            context.lineTo(end.x * width, end.y * height)
            context.stroke()
        }

        for (const point of landmarks) {
            context.beginPath()
            context.arc(point.x * width, point.y * height, 3, 0, Math.PI * 2)
            context.fill()
        }
    }, [])

    const sendPrediction = useCallback(async () => {
        const imageData = captureFrame()
        if (!imageData) {
            return
        }

        setIsSending(true)
        setErrorMessage(null)

        try {
            const controller = new AbortController()
            const timeoutId = window.setTimeout(() => controller.abort(), PREDICT_TIMEOUT_MS)
            const response = await fetch(`${API_BASE}/predict`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    session_id: sessionId,
                    mode,
                    image_data: imageData,
                }),
                signal: controller.signal,
            }).finally(() => {
                window.clearTimeout(timeoutId)
            })

            if (!response.ok) {
                throw new Error(`Predict request failed with status ${response.status}`)
            }

            const data = (await response.json()) as PredictResponse
            setServerOnline(true)
            setHandDetected(data.hand_detected)
            setCurrentPrediction(data.current_prediction ?? "-")
            setConfidence(data.confidence)
            setSentence(data.sentence)
            setBufferProgress(data.buffer_progress)
            setSequenceProgress(data.sequence_progress)
            drawLandmarks(normalizeLandmarkPayload(data.landmarks ?? []))
        } catch {
            setServerOnline(false)
            setErrorMessage("Prediction failed. Check if backend is running and reachable.")
            drawLandmarks([])
        } finally {
            setIsSending(false)
        }
    }, [captureFrame, drawLandmarks, mode, sessionId])

    useEffect(() => {
        if (!isCameraOn) {
            return
        }

        const timer = window.setInterval(() => {
            if (inFlightRef.current) {
                return
            }

            inFlightRef.current = true
            void sendPrediction().finally(() => {
                inFlightRef.current = false
            })
        }, PREDICT_INTERVAL_MS)

        return () => {
            window.clearInterval(timer)
            inFlightRef.current = false
        }
    }, [isCameraOn, sendPrediction])

    const confidenceText =
        confidence === null ? "-" : `${Math.max(0, Math.min(100, confidence * 100)).toFixed(1)}%`

    return (
        <div className="isl-shell px-4 py-8 md:px-8 md:py-10">
            <main className="mx-auto flex w-full max-w-6xl flex-col gap-6">
                <header className="glass-card p-5 md:p-7">
                    <p className="mb-2 text-xs font-semibold tracking-[0.2em] text-slate-700 uppercase">
                        Indian Sign Language Assist
                    </p>
                    <h1 className="mb-3 text-2xl font-bold text-slate-900 md:text-4xl" style={{ fontFamily: "var(--font-display)" }}>
                        Real-time Sign to Text Bridge
                    </h1>
                    <p className="max-w-3xl text-sm text-slate-700 md:text-base">
                        Live webcam inference for static gestures and motion sequences, directly integrated with your trained models.
                    </p>
                    <div className="mt-4 flex flex-wrap gap-2 text-xs md:text-sm">
                        <span
                            className={`status-chip ${serverOnline === null ? "" : serverOnline ? "status-ok" : "status-bad"
                                }`}
                        >
                            API: {serverOnline === null ? "Checking..." : serverOnline ? "Online" : "Offline"}
                        </span>
                        <span className="status-chip">Session: {sessionId.slice(0, 8)}</span>
                        <span className="status-chip">Frame Tx: {isSending ? "Running" : "Idle"}</span>
                    </div>
                </header>

                <section className="grid gap-6 lg:grid-cols-[1.35fr_1fr]">
                    <article className="glass-card p-4 md:p-6">
                        <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
                            <h2 className="text-lg font-semibold text-slate-900 md:text-xl" style={{ fontFamily: "var(--font-display)" }}>
                                Camera Feed
                            </h2>
                            <div className="flex flex-wrap gap-2">
                                <Button onClick={() => void startCamera()} disabled={isCameraOn}>
                                    Start Camera
                                </Button>
                                <Button variant="outline" onClick={stopCamera} disabled={!isCameraOn}>
                                    Stop Camera
                                </Button>
                            </div>
                        </div>

                        <div className="camera-frame relative overflow-hidden rounded-xl">
                            <video
                                ref={videoRef}
                                className="h-full w-full object-cover"
                                style={{ transform: "scaleX(-1)" }}
                                muted
                                playsInline
                            />
                            <canvas
                                ref={overlayCanvasRef}
                                className="pointer-events-none absolute inset-0 h-full w-full"
                            />
                            {!isCameraOn && (
                                <div className="absolute inset-0 flex items-center justify-center bg-slate-950/70 p-6 text-center text-sm text-white">
                                    Start camera to begin inference.
                                </div>
                            )}
                            <canvas ref={captureCanvasRef} className="hidden" />
                        </div>

                        <div className="mt-4 flex flex-wrap gap-2">
                            <Button
                                variant={mode === "static" ? "default" : "outline"}
                                onClick={() => setMode("static")}
                            >
                                Static Mode
                            </Button>
                            <Button
                                variant={mode === "motion" ? "default" : "outline"}
                                onClick={() => setMode("motion")}
                            >
                                Motion Mode
                            </Button>
                        </div>
                        <p className="mt-3 text-xs text-slate-600 md:text-sm">
                            {mode === "static"
                                ? "Static mode uses a 10-frame stability buffer to reduce jitter before appending characters."
                                : "Motion mode waits for 30 landmark frames before classifying a temporal action."}
                        </p>
                    </article>

                    <article className="glass-card p-4 md:p-6">
                        <h2 className="mb-4 text-lg font-semibold text-slate-900 md:text-xl" style={{ fontFamily: "var(--font-display)" }}>
                            Live Output
                        </h2>

                        <div className="metric-grid mb-4">
                            <div className="metric-card">
                                <p className="metric-label">Prediction</p>
                                <p className="metric-value">{currentPrediction}</p>
                            </div>
                            <div className="metric-card">
                                <p className="metric-label">Confidence</p>
                                <p className="metric-value">{confidenceText}</p>
                            </div>
                            <div className="metric-card">
                                <p className="metric-label">Hand</p>
                                <p className="metric-value">{handDetected ? "Detected" : "Not Detected"}</p>
                            </div>
                            <div className="metric-card">
                                <p className="metric-label">Progress</p>
                                <p className="metric-value">{mode === "static" ? `${bufferProgress}/10` : `${sequenceProgress}/30`}</p>
                            </div>
                        </div>

                        <div className="rounded-xl border border-slate-300 bg-white/85 p-3">
                            <p className="mb-2 text-xs font-semibold tracking-wide text-slate-700 uppercase">Sentence</p>
                            <div className="min-h-24 rounded-lg bg-slate-900 px-3 py-2 font-mono text-sm leading-relaxed text-emerald-300">
                                {sentence || "..."}
                            </div>
                        </div>

                        <div className="mt-4 flex flex-wrap gap-2">
                            <Button variant="outline" onClick={() => void sendControl("clear")}>
                                Clear
                            </Button>
                            <Button variant="outline" onClick={() => void sendControl("backspace")}>
                                Backspace
                            </Button>
                            <Button variant="outline" onClick={() => void sendControl("delete_word")}>
                                Delete Word
                            </Button>
                        </div>

                        {errorMessage && (
                            <p className="mt-4 rounded-lg border border-red-300 bg-red-50 px-3 py-2 text-sm text-red-700">
                                {errorMessage}
                            </p>
                        )}
                    </article>
                </section>
            </main>
        </div>
    )
}
