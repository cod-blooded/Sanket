"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import {
    ArrowTurnBackwardIcon,
    CameraVideoIcon,
    Delete02Icon,
    KeyboardIcon,
    MaximizeScreenIcon,
    MinimizeScreenIcon,
    PlayIcon,
    ServerStack03Icon,
    VideoOffIcon,
    VolumeHighIcon,
    WasteIcon,
} from "@hugeicons/core-free-icons"
import { HugeiconsIcon } from "@hugeicons/react"

import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

type Mode = "static" | "motion"
type ControlAction =
    | "clear"
    | "backspace"
    | "delete_word"
    | "reset_session"
    | "reset_tracking"

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
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 4],
    [0, 5],
    [5, 6],
    [6, 7],
    [7, 8],
    [5, 9],
    [9, 10],
    [10, 11],
    [11, 12],
    [9, 13],
    [13, 14],
    [14, 15],
    [15, 16],
    [13, 17],
    [17, 18],
    [18, 19],
    [19, 20],
    [0, 17],
]

const STATIC_CAPTURE_WIDTH = 640
const MOTION_CAPTURE_WIDTH = 420
const STATIC_INTERVAL_MS = 220
const MOTION_INTERVAL_MS = 80
const PREDICT_TIMEOUT_MS = 8000
const BOOTSTRAP_DELAY_MS = 2600

const API_BASE =
    process.env.NEXT_PUBLIC_INFERENCE_API_URL?.replace(/\/$/, "") ??
    "http://127.0.0.1:8000"

const glassSurface =
    "border border-white/15 bg-[rgb(5_12_23_/_50%)] shadow-[0_1rem_2.5rem_rgb(0_0_0_/_24%)] backdrop-blur-[16px]"
const livePillClass = cn(
    glassSurface,
    "inline-flex min-h-[2.25rem] items-center gap-[0.42rem] rounded-full px-[0.7rem] py-[0.45rem] text-[0.78rem] font-bold whitespace-nowrap text-[rgb(226_236_246)] max-[640px]:min-h-8 max-[640px]:px-[0.56rem] max-[640px]:py-[0.38rem] max-[640px]:text-[0.72rem]"
)
const dockButtonClass = cn(
    glassSurface,
    "inline-flex min-h-[2.35rem] cursor-pointer items-center justify-center gap-[0.42rem] rounded-full px-[0.72rem] py-[0.48rem] text-[0.82rem] font-extrabold text-[rgb(230_239_247)] transition-[transform,background-color] duration-200 hover:-translate-y-px max-[640px]:min-w-0 max-[640px]:px-[0.35rem] max-[640px]:py-2 max-[640px]:text-[0.72rem]"
)
const metricStripClass = cn(
    glassSurface,
    "grid gap-[0.3rem] rounded-[0.85rem] p-3 max-[900px]:p-[0.62rem] max-[620px]:landscape:p-[0.55rem] max-[640px]:[&:nth-child(4)]:hidden"
)
const miniBarClass =
    "h-[0.34rem] overflow-hidden rounded-full bg-white/15 [&>i]:block [&>i]:h-full [&>i]:rounded-[inherit] [&>i]:bg-[linear-gradient(90deg,rgb(36_214_190),rgb(255_205_77))] [&>i]:transition-[width] [&>i]:duration-300"

function normalizeLandmarkPayload(
    points: LandmarkPayloadPoint[]
): LandmarkPoint[] {
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

function clampPercent(value: number) {
    return Math.max(0, Math.min(100, value))
}

function pickHumanVoice(voices: SpeechSynthesisVoice[]) {
    const englishVoices = voices.filter((voice) =>
        voice.lang.toLowerCase().startsWith("en")
    )
    const candidates = englishVoices.length ? englishVoices : voices
    const preferredNameParts = [
        "natural",
        "online",
        "aria",
        "jenny",
        "guy",
        "sonia",
        "google",
        "microsoft",
    ]

    return (
        candidates.find((voice) =>
            preferredNameParts.some((part) =>
                voice.name.toLowerCase().includes(part)
            )
        ) ??
        candidates.find((voice) => voice.localService) ??
        candidates[0] ??
        null
    )
}

export default function Page() {
    const [sessionId, setSessionId] = useState<string | null>(null)

    const viewerRef = useRef<HTMLDivElement | null>(null)
    const videoRef = useRef<HTMLVideoElement | null>(null)
    const captureCanvasRef = useRef<HTMLCanvasElement | null>(null)
    const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null)
    const transcriptScrollRef = useRef<HTMLDivElement | null>(null)
    const shouldAutoScrollTranscriptRef = useRef(true)
    const inFlightRef = useRef(false)
    const handDetectedRef = useRef(false)
    const speechVoiceRef = useRef<SpeechSynthesisVoice | null>(null)

    const [isBootstrapping, setIsBootstrapping] = useState(true)
    const [mode, setMode] = useState<Mode>("static")
    const [isCameraOn, setIsCameraOn] = useState(false)
    const [isDetecting, setIsDetecting] = useState(false)
    const [isFullscreen, setIsFullscreen] = useState(false)
    const [serverOnline, setServerOnline] = useState<boolean | null>(null)

    const [handDetected, setHandDetected] = useState(false)
    const [handWakeKey, setHandWakeKey] = useState(0)
    const [currentPrediction, setCurrentPrediction] = useState("-")
    const [confidence, setConfidence] = useState<number | null>(null)
    const [sentence, setSentence] = useState("")
    const [speechSupported, setSpeechSupported] = useState(false)
    const [isSpeaking, setIsSpeaking] = useState(false)
    const [bufferProgress, setBufferProgress] = useState(0)
    const [sequenceProgress, setSequenceProgress] = useState(0)
    const [errorMessage, setErrorMessage] = useState<string | null>(null)

    useEffect(() => {
        setSessionId(createSessionId())
    }, [])

    useEffect(() => {
        const timer = window.setTimeout(() => {
            setIsBootstrapping(false)
        }, BOOTSTRAP_DELAY_MS)

        return () => {
            window.clearTimeout(timer)
        }
    }, [])

    useEffect(() => {
        if (typeof window === "undefined" || !("speechSynthesis" in window)) {
            setSpeechSupported(false)
            return
        }

        const loadVoices = () => {
            speechVoiceRef.current = pickHumanVoice(
                window.speechSynthesis.getVoices()
            )
            setSpeechSupported(true)
        }

        loadVoices()
        window.speechSynthesis.addEventListener("voiceschanged", loadVoices)

        return () => {
            window.speechSynthesis.cancel()
            window.speechSynthesis.removeEventListener("voiceschanged", loadVoices)
        }
    }, [])

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

        const timer = window.setInterval(() => {
            void checkHealth()
        }, 10000)

        return () => {
            window.clearInterval(timer)
        }
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
        setIsDetecting(false)
        handDetectedRef.current = false
        setHandDetected(false)
    }, [])

    const startCamera = useCallback(async () => {
        setErrorMessage(null)

        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: "user",
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                },
                audio: false,
            })

            if (!videoRef.current) {
                return false
            }

            videoRef.current.srcObject = stream
            await videoRef.current.play()
            setIsCameraOn(true)
            return true
        } catch {
            setIsCameraOn(false)
            setErrorMessage("Camera access is needed to show the live feed.")
            return false
        }
    }, [])

    useEffect(() => {
        if (isBootstrapping) {
            return
        }

        void startCamera()
    }, [isBootstrapping, startCamera])

    useEffect(() => {
        return () => {
            stopCamera()
        }
    }, [stopCamera])

    const sendControl = useCallback(
        async (action: ControlAction) => {
            if (!sessionId) {
                return
            }

            try {
                const response = await fetch(`${API_BASE}/control`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ session_id: sessionId, action }),
                })

                if (!response.ok) {
                    throw new Error(
                        `Control request failed with status ${response.status}`
                    )
                }

                const data = (await response.json()) as { sentence: string }
                setSentence(data.sentence)
            } catch {
                setErrorMessage("Failed to send control action to backend.")
            }
        },
        [sessionId]
    )

    const speakTranscript = useCallback(() => {
        const text = sentence.trim()

        if (!speechSupported || typeof window === "undefined") {
            setErrorMessage("Text-to-speech is not available in this browser.")
            return
        }

        if (window.speechSynthesis.speaking) {
            window.speechSynthesis.cancel()
            setIsSpeaking(false)
            return
        }

        if (!text) {
            setErrorMessage("There is no transcription to read yet.")
            return
        }

        setErrorMessage(null)
        const utterance = new SpeechSynthesisUtterance(text)
        utterance.voice = speechVoiceRef.current
        utterance.rate = 0.92
        utterance.pitch = 1
        utterance.volume = 1
        utterance.onstart = () => setIsSpeaking(true)
        utterance.onend = () => setIsSpeaking(false)
        utterance.onerror = () => {
            setIsSpeaking(false)
            setErrorMessage("Could not play the transcription voice.")
        }

        window.speechSynthesis.cancel()
        window.speechSynthesis.speak(utterance)
    }, [sentence, speechSupported])

    const resetLocalTracking = useCallback(() => {
        handDetectedRef.current = false
        setHandDetected(false)
        setCurrentPrediction("-")
        setConfidence(null)
        setBufferProgress(0)
        setSequenceProgress(0)

        const overlay = overlayCanvasRef.current
        if (overlay) {
            const context = overlay.getContext("2d")
            context?.clearRect(0, 0, overlay.width, overlay.height)
        }
    }, [])

    const changeMode = useCallback(
        (nextMode: Mode) => {
            setMode(nextMode)
            resetLocalTracking()
            void sendControl("reset_tracking")
        },
        [resetLocalTracking, sendControl]
    )

    const startDetection = useCallback(async () => {
        const cameraReady = isCameraOn || (await startCamera())
        if (!cameraReady) {
            return
        }

        setErrorMessage(null)
        resetLocalTracking()
        await sendControl("reset_tracking")
        setIsDetecting(true)
    }, [isCameraOn, resetLocalTracking, sendControl, startCamera])

    const toggleDetection = useCallback(async () => {
        if (isDetecting) {
            setIsDetecting(false)
            return
        }

        await startDetection()
    }, [isDetecting, startDetection])

    const toggleFullscreen = useCallback(async () => {
        try {
            if (document.fullscreenElement) {
                await document.exitFullscreen()
                return
            }

            await viewerRef.current?.requestFullscreen()
        } catch {
            setErrorMessage("Fullscreen is not available in this browser window.")
        }
    }, [])

    useEffect(() => {
        const onFullscreenChange = () => {
            setIsFullscreen(Boolean(document.fullscreenElement))
        }

        document.addEventListener("fullscreenchange", onFullscreenChange)
        return () => {
            document.removeEventListener("fullscreenchange", onFullscreenChange)
        }
    }, [])

    useEffect(() => {
        const onKeyDown = (event: KeyboardEvent) => {
            const target = event.target as HTMLElement | null
            if (
                target &&
                (target.tagName === "INPUT" ||
                    target.tagName === "TEXTAREA" ||
                    target.tagName === "SELECT" ||
                    target.isContentEditable)
            ) {
                return
            }

            const key = event.key.toLowerCase()
            if (key === " " || key === "enter") {
                event.preventDefault()
                void toggleDetection()
                return
            }

            if (key === "f") {
                event.preventDefault()
                void toggleFullscreen()
                return
            }

            if (key === "m") {
                event.preventDefault()
                changeMode("motion")
                return
            }

            if (key === "n") {
                event.preventDefault()
                changeMode("static")
                return
            }

            if (key === "c") {
                event.preventDefault()
                void sendControl("clear")
                return
            }

            if (key === "b") {
                event.preventDefault()
                void sendControl("backspace")
                return
            }

            if (key === "w") {
                event.preventDefault()
                void sendControl("delete_word")
            }
        }

        window.addEventListener("keydown", onKeyDown)
        return () => {
            window.removeEventListener("keydown", onKeyDown)
        }
    }, [changeMode, sendControl, toggleDetection, toggleFullscreen])

    const captureFrame = useCallback(() => {
        const video = videoRef.current
        const canvas = captureCanvasRef.current

        if (
            !video ||
            !canvas ||
            video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA
        ) {
            return null
        }

        const width = video.videoWidth
        const height = video.videoHeight

        if (!width || !height) {
            return null
        }

        const preferredWidth =
            mode === "motion" ? MOTION_CAPTURE_WIDTH : STATIC_CAPTURE_WIDTH
        const targetWidth = Math.min(preferredWidth, width)
        const targetHeight = Math.round((height / width) * targetWidth)

        canvas.width = targetWidth
        canvas.height = targetHeight

        const context = canvas.getContext("2d")
        if (!context) {
            return null
        }

        const quality = mode === "motion" ? 0.5 : 0.6
        context.drawImage(video, 0, 0, targetWidth, targetHeight)
        return canvas.toDataURL("image/jpeg", quality)
    }, [mode])

    const drawLandmarks = useCallback((landmarks: LandmarkPoint[]) => {
        const video = videoRef.current
        const canvas = overlayCanvasRef.current

        if (
            !video ||
            !canvas ||
            video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA
        ) {
            return
        }

        const width = video.videoWidth
        const height = video.videoHeight
        if (!width || !height) {
            return
        }

        const bounds = canvas.getBoundingClientRect()
        const displayWidth = bounds.width
        const displayHeight = bounds.height
        const pixelRatio = window.devicePixelRatio || 1
        const canvasWidth = Math.round(displayWidth * pixelRatio)
        const canvasHeight = Math.round(displayHeight * pixelRatio)

        if (!displayWidth || !displayHeight) {
            return
        }

        if (canvas.width !== canvasWidth || canvas.height !== canvasHeight) {
            canvas.width = canvasWidth
            canvas.height = canvasHeight
        }

        const context = canvas.getContext("2d")
        if (!context) {
            return
        }

        context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0)
        context.clearRect(0, 0, displayWidth, displayHeight)

        if (!landmarks.length) {
            return
        }

        const scale = Math.max(displayWidth / width, displayHeight / height)
        const renderedWidth = width * scale
        const renderedHeight = height * scale
        const offsetX = (displayWidth - renderedWidth) / 2
        const offsetY = (displayHeight - renderedHeight) / 2
        const toDisplayPoint = (point: LandmarkPoint) => ({
            x: offsetX + point.x * renderedWidth,
            y: offsetY + point.y * renderedHeight,
        })

        context.lineWidth = 3
        context.strokeStyle = "rgba(36, 214, 190, 0.94)"
        context.fillStyle = "rgba(255, 205, 77, 0.96)"
        context.lineCap = "round"
        context.lineJoin = "round"

        for (const [startIdx, endIdx] of HAND_CONNECTIONS) {
            const start = landmarks[startIdx]
            const end = landmarks[endIdx]

            if (!start || !end) {
                continue
            }

            const displayStart = toDisplayPoint(start)
            const displayEnd = toDisplayPoint(end)
            context.beginPath()
            context.moveTo(displayStart.x, displayStart.y)
            context.lineTo(displayEnd.x, displayEnd.y)
            context.stroke()
        }

        for (const point of landmarks) {
            const displayPoint = toDisplayPoint(point)
            context.beginPath()
            context.arc(displayPoint.x, displayPoint.y, 4, 0, Math.PI * 2)
            context.fill()
        }
    }, [])

    const sendPrediction = useCallback(async () => {
        if (!sessionId) {
            return
        }

        const imageData = captureFrame()
        if (!imageData) {
            return
        }

        setErrorMessage(null)

        try {
            const controller = new AbortController()
            const timeoutId = window.setTimeout(
                () => controller.abort(),
                PREDICT_TIMEOUT_MS
            )
            const response = await fetch(`${API_BASE}/predict`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    session_id: sessionId,
                    mode,
                    image_data: imageData,
                    flip_horizontal: true,
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
            if (data.hand_detected && !handDetectedRef.current) {
                setHandWakeKey((key) => key + 1)
            }
            handDetectedRef.current = data.hand_detected
            setHandDetected(data.hand_detected)
            setCurrentPrediction(data.current_prediction ?? "-")
            setConfidence(data.confidence)
            setSentence(data.sentence)
            setBufferProgress(data.buffer_progress)
            setSequenceProgress(data.sequence_progress)
            drawLandmarks(normalizeLandmarkPayload(data.landmarks ?? []))
        } catch {
            setServerOnline(false)
            setErrorMessage(
                "Prediction failed. Check if backend is running and reachable."
            )
            handDetectedRef.current = false
            setHandDetected(false)
            drawLandmarks([])
        }
    }, [captureFrame, drawLandmarks, mode, sessionId])

    useEffect(() => {
        if (!isCameraOn || !isDetecting) {
            return
        }

        const timer = window.setInterval(
            () => {
                if (inFlightRef.current) {
                    return
                }

                inFlightRef.current = true
                void sendPrediction().finally(() => {
                    inFlightRef.current = false
                })
            },
            mode === "motion" ? MOTION_INTERVAL_MS : STATIC_INTERVAL_MS
        )

        return () => {
            window.clearInterval(timer)
            inFlightRef.current = false
        }
    }, [isCameraOn, isDetecting, mode, sendPrediction])

    const confidenceValue =
        confidence === null ? 0 : clampPercent(confidence * 100)
    const confidenceText =
        confidence === null ? "-" : `${confidenceValue.toFixed(1)}%`
    const progressValue = mode === "static" ? bufferProgress : sequenceProgress
    const progressTotal = mode === "static" ? 10 : 30
    const progressPercent = clampPercent((progressValue / progressTotal) * 100)
    const sentenceText =
        sentence.trim() || (isDetecting ? "detecting..." : "Ready")
    const liveEffectActive = isDetecting && handDetected

    useEffect(() => {
        const transcript = transcriptScrollRef.current
        if (!transcript) {
            return
        }

        if (shouldAutoScrollTranscriptRef.current) {
            transcript.scrollTop = transcript.scrollHeight
        }
    }, [sentenceText])

    if (isBootstrapping) {
        return (
            <main
                className="grid min-h-svh place-items-center bg-[linear-gradient(180deg,rgb(6_15_28),rgb(3_7_15))] p-6"
                aria-label="Loading Sanket"
            >
                <div className="grid aspect-square w-[min(22rem,76vw)] animate-in place-items-center rounded-[1.25rem] duration-700 zoom-in-95 fade-in slide-in-from-bottom-3 max-[640px]:rounded-[0.9rem]">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                        className="block size-full rounded-[inherit]"
                        src="/sanket-loader.svg"
                        alt="Sanket"
                    />
                </div>
            </main>
        )
    }

    return (
        <main className="min-h-svh bg-[#050915]">
            <section
                ref={viewerRef}
                className="relative isolate min-h-svh overflow-hidden bg-[linear-gradient(120deg,rgb(8_18_32),rgb(5_12_21))] text-white [&:fullscreen]:h-screen [&:fullscreen]:min-h-screen [&:fullscreen]:w-screen"
                aria-label="Sign detection camera"
            >
                <video
                    ref={videoRef}
                    className={cn(
                        "absolute inset-0 z-0 size-full -scale-x-100 bg-[#07101c] object-cover transition-[opacity,filter,transform] duration-[420ms]",
                        liveEffectActive
                            ? "opacity-100 brightness-[1.05] contrast-[1.08] saturate-[1.16]"
                            : isDetecting
                                ? "opacity-100 brightness-[0.96] contrast-[1.04] saturate-[1.04]"
                                : "opacity-50 brightness-[0.78] contrast-[0.92] saturate-[0.72]"
                    )}
                    muted
                    playsInline
                />
                <canvas
                    ref={overlayCanvasRef}
                    className="pointer-events-none absolute inset-0 z-[1] size-full object-cover"
                />
                <canvas ref={captureCanvasRef} className="hidden" />

                <div className="pointer-events-none absolute inset-0 z-[2] bg-[linear-gradient(90deg,rgb(0_0_0_/_42%),transparent_28%,transparent_68%,rgb(0_0_0_/_26%)),linear-gradient(180deg,rgb(0_0_0_/_55%),transparent_22%,transparent_58%,rgb(0_0_0_/_28%))]" />
                <div className="pointer-events-none absolute inset-0 z-[3] bg-[linear-gradient(180deg,transparent_34%,rgb(1_5_12_/_24%)_54%,rgb(1_5_12_/_78%)_82%,rgb(1_5_12_/_96%)_100%)]" />
                <div
                    className={cn(
                        "pointer-events-none absolute inset-0 z-[4] transition-opacity duration-500",
                        liveEffectActive ? "opacity-100" : "opacity-0"
                    )}
                >
                    <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_42%,rgb(36_214_190_/_18%),transparent_32%),linear-gradient(120deg,rgb(36_214_190_/_14%),transparent_24%,transparent_72%,rgb(255_205_77_/_12%))]" />
                    <div className="absolute inset-0 shadow-[inset_0_0_3rem_rgb(36_214_190_/_22%),inset_0_0_7rem_rgb(255_205_77_/_10%)]" />
                </div>
                {!isCameraOn && (
                    <div className="absolute top-1/2 left-1/2 z-[5] grid w-[min(25rem,calc(100vw-2rem))] -translate-x-1/2 -translate-y-1/2 justify-items-center gap-4 rounded-[1.1rem] border border-white/15 bg-[rgb(6_14_26_/_72%)] p-5 text-center shadow-[0_1.5rem_4rem_rgb(0_0_0_/_34%)] backdrop-blur-[18px]">
                        <span className="grid size-16 place-items-center rounded-full bg-[rgb(36_214_190_/_12%)] text-[rgb(36_214_190)]">
                            <HugeiconsIcon
                                icon={CameraVideoIcon}
                                size={34}
                                strokeWidth={1.7}
                            />
                        </span>
                        <p className="m-0 text-[0.96rem] leading-[1.45] font-bold text-[rgb(230_239_247)]">
                            Allow camera access to open the live view.
                        </p>
                        <Button
                            className="min-h-[2.55rem] gap-2 rounded-full bg-[rgb(36_214_190)] text-[rgb(4_13_23)] shadow-[0_0.75rem_2rem_rgb(36_214_190_/_22%)] hover:bg-[rgb(36_214_190)]"
                            onClick={() => void startCamera()}
                        >
                            <HugeiconsIcon icon={CameraVideoIcon} size={18} />
                            Enable camera
                        </Button>
                    </div>
                )}

                <div className="absolute top-[max(1rem,env(safe-area-inset-top))] right-[max(1rem,env(safe-area-inset-right))] left-[max(1rem,env(safe-area-inset-left))] z-[5] flex items-start justify-between gap-4 max-[900px]:flex-col max-[900px]:items-stretch max-[640px]:top-[max(0.75rem,env(safe-area-inset-top))] max-[640px]:right-3 max-[640px]:left-3 max-[640px]:gap-[0.7rem]">
                    <div className="hidden sm:flex flex-wrap items-center gap-[0.55rem] max-[640px]:gap-[0.42rem]">
                        <span
                            className={cn(
                                livePillClass,
                                serverOnline
                                    ? "border-[rgb(44_214_190_/_34%)] bg-[rgb(18_141_123_/_28%)] text-[rgb(214_255_248)]"
                                    : serverOnline === false
                                        ? "border-[rgb(248_92_92_/_36%)] bg-[rgb(137_24_42_/_34%)] text-[rgb(255_224_227)]"
                                        : ""
                            )}
                        >
                            <HugeiconsIcon icon={ServerStack03Icon} size={16} />
                            {serverOnline === null
                                ? "Checking server"
                                : serverOnline
                                    ? "Server live"
                                    : "Server offline"}
                        </span>
                        <span className={cn(livePillClass, "max-[640px]:hidden")}>
                            <HugeiconsIcon icon={KeyboardIcon} size={16} />
                            Space start, F fullscreen
                        </span>
                        {isDetecting && (
                            <div
                                className={cn(
                                    livePillClass,
                                    "max-w-[calc(100vw-2rem)] max-[640px]:top-[3.75rem] max-[640px]:left-3",
                                    liveEffectActive &&
                                    "border-[rgb(36_214_190_/_48%)] bg-[rgb(8_60_58_/_48%)] text-[rgb(226_255_250)] shadow-[0_0_2.2rem_rgb(36_214_190_/_28%)]"
                                )}
                            >
                                <span
                                    className={cn(
                                        "size-2 rounded-full bg-[rgb(177_197_216)]",
                                        liveEffectActive &&
                                        "animate-pulse bg-[rgb(36_214_190)] shadow-[0_0_1rem_rgb(36_214_190_/_95%)]"
                                    )}
                                />
                                {liveEffectActive
                                    ? currentPrediction !== "-"
                                        ? currentPrediction
                                        : "Hand locked"
                                    : "Seeking hand"}
                            </div>
                        )}
                    </div>

                    <div className="flex flex-wrap items-center gap-[0.55rem] max-[900px]:justify-between">
                        <div
                            className={cn(
                                glassSurface,
                                "grid h-[2.4rem] w-44 grid-cols-2 rounded-full p-[0.2rem] max-[640px]:h-9 max-[640px]:w-[min(10.5rem,calc(100vw-5rem))]"
                            )}
                            aria-label="Recognition mode"
                        >
                            <button
                                className={cn(
                                    "cursor-pointer rounded-full border-0 bg-transparent text-[0.8rem] font-extrabold text-[rgb(210_225_238)] transition-colors duration-200",
                                    mode === "static" &&
                                    "bg-[rgb(246_250_252)] text-[rgb(4_13_23)]"
                                )}
                                onClick={() => changeMode("static")}
                                type="button"
                            >
                                Static
                            </button>
                            <button
                                className={cn(
                                    "cursor-pointer rounded-full border-0 bg-transparent text-[0.8rem] font-extrabold text-[rgb(210_225_238)] transition-colors duration-200",
                                    mode === "motion" &&
                                    "bg-[rgb(246_250_252)] text-[rgb(4_13_23)]"
                                )}
                                onClick={() => changeMode("motion")}
                                type="button"
                            >
                                Motion
                            </button>
                        </div>
                        <button
                            className={cn(
                                glassSurface,
                                "inline-grid size-[2.4rem] cursor-pointer place-items-center rounded-full border-[rgb(36_214_190_/_28%)] bg-[rgb(6_20_32_/_68%)] text-[rgb(231_255_251)] transition-[transform,background-color] duration-200 hover:-translate-y-px"
                            )}
                            onClick={() => void toggleFullscreen()}
                            type="button"
                            aria-label={isFullscreen ? "Exit fullscreen" : "Enter fullscreen"}
                            title={isFullscreen ? "Exit fullscreen" : "Enter fullscreen"}
                        >
                            <HugeiconsIcon
                                icon={isFullscreen ? MinimizeScreenIcon : MaximizeScreenIcon}
                                size={21}
                            />
                        </button>
                    </div>
                </div>


                {!isDetecting && isCameraOn && (
                    <div className="absolute top-1/3 sm:top-1/2 left-1/2 z-100 flex max-w-[calc(100vw-2rem)] -translate-x-1/2 -translate-y-1/2 items-center gap-4 rounded-full border border-white/15 bg-[rgb(4_12_22_/_58%)] py-[0.8rem] pr-5 pl-[0.8rem] shadow-[0_1.5rem_4rem_rgb(0_0_0_/_34%)] backdrop-blur-[18px] max-[640px]:w-[calc(100vw-1.5rem)] max-[640px]:rounded-2xl max-[640px]:p-[0.72rem]">
                        <button
                            className="grid size-[4.25rem] shrink-0 cursor-pointer place-items-center rounded-full border-0 bg-[linear-gradient(135deg,rgb(36_214_190),rgb(255_205_77))] text-[rgb(5_13_22)] shadow-[0_1rem_2rem_rgb(36_214_190_/_26%)] transition-transform duration-200 hover:-translate-y-px max-[640px]:size-14"
                            onClick={() => void startDetection()}
                            type="button"
                        >
                            <HugeiconsIcon icon={PlayIcon} size={34} fill="currentColor" />
                        </button>
                        <div>
                            <p className="m-0 text-[0.78rem] font-extrabold text-[rgb(185_210_226)] uppercase">
                                Sanket is ready
                            </p>
                            <p className="m-0 font-heading text-[clamp(1.2rem,2.4vw,1.8rem)] leading-[1.05] font-extrabold text-white max-[640px]:whitespace-normal min-[641px]:whitespace-nowrap">
                                Start sign detection
                            </p>
                        </div>
                    </div>
                )}

                <div
                    className="hidden absolute top-[5.2rem] right-[max(1rem,env(safe-area-inset-right))] z-[5] md:grid w-[min(18rem,calc(100vw-2rem))] gap-[0.55rem] max-[900px]:top-[9.1rem] max-[900px]:right-auto max-[900px]:left-[max(1rem,env(safe-area-inset-left))] max-[900px]:w-[min(30rem,calc(100vw-2rem))] max-[900px]:grid-cols-2 max-[640px]:top-[8.1rem] max-[640px]:right-3 max-[640px]:left-3 max-[640px]:w-auto max-[620px]:landscape:top-[4.6rem] max-[620px]:landscape:w-[min(42rem,calc(100vw-2rem))] max-[620px]:landscape:grid-cols-4"
                    aria-label="Inference metrics"
                >
                    {/* <div className={metricStripClass}>
            <span className="text-[0.68rem] font-extrabold text-[rgb(177_197_216)] uppercase">
              Prediction
            </span>
            <strong className="font-heading text-[1.05rem] leading-none [overflow-wrap:anywhere] text-white">
              {currentPrediction}
            </strong>
          </div> */}
                    <div className={metricStripClass}>
                        <span className="text-[0.68rem] font-extrabold text-[rgb(177_197_216)] uppercase">
                            Confidence
                        </span>
                        <strong className="font-heading text-[1.05rem] leading-none [overflow-wrap:anywhere] text-white">
                            {confidenceText}
                        </strong>
                        <div className={miniBarClass}>
                            <i style={{ width: `${confidenceValue}%` }} />
                        </div>
                    </div>
                    <div className={metricStripClass}>
                        <span className="text-[0.68rem] font-extrabold text-[rgb(177_197_216)] uppercase">
                            Progress
                        </span>
                        <strong className="font-heading text-[1.05rem] leading-none wrap-anywhere text-white">
                            {progressValue}/{progressTotal}
                        </strong>
                        <div className={miniBarClass}>
                            <i style={{ width: `${progressPercent}%` }} />
                        </div>
                    </div>
                    {/* <div className={metricStripClass}>
            <span className="text-[0.68rem] font-extrabold text-[rgb(177_197_216)] uppercase">
              Hand
            </span>
            <strong className="font-heading text-[1.05rem] leading-none [overflow-wrap:anywhere] text-white">
              {handDetected ? "Detected" : "Waiting"}
            </strong>
          </div> */}
                </div>

                <div className="absolute right-[max(1rem,env(safe-area-inset-right))] bottom-[max(1rem,env(safe-area-inset-bottom))] left-[max(1rem,env(safe-area-inset-left))] z-[5] grid max-h-[78vh] gap-[0.9rem] overflow-hidden max-[640px]:right-3 max-[640px]:bottom-[max(0.75rem,env(safe-area-inset-bottom))] max-[640px]:left-3 max-[640px]:max-h-[36vh]">
                    {errorMessage && (
                        <p
                            className={cn(
                                glassSurface,
                                "m-0 justify-self-start rounded-full border-[rgb(248_92_92_/_34%)] bg-[rgb(137_24_42_/_42%)] px-[0.86rem] py-[0.62rem] text-[0.82rem] font-bold text-[rgb(255_224_227)]"
                            )}
                        >
                            {errorMessage}
                        </p>
                    )}

                    <div className="relative h-[70vh] min-h-[7.2rem] w-[min(35rem,calc(100vw-2rem))] max-w-full overflow-hidden max-[640px]:h-[22vh] max-[640px]:max-h-42 max-[640px]:min-h-[6.2rem] max-[640px]:w-full max-[620px]:landscape:h-[22vh] max-[620px]:landscape:max-h-[7.8rem] max-[620px]:landscape:min-h-[4.7rem]">
                        {/* <p className="absolute top-0 left-0 z-1 m-0 px-0 pt-[0.15rem] pb-[0.4rem] text-[0.76rem] font-black text-[rgb(169_195_215)] uppercase [text-shadow:0_0.4rem_1.1rem_rgb(0_0_0/72%)]">
              {isDetecting ? "Detected text" : "Transcript"}
            </p> */}
                        <div
                            ref={transcriptScrollRef}
                            className="absolute flex items-end inset-x-0 top-[1.35rem] h-[calc(100%-1.35rem)] min-h-0 w-full overflow-x-hidden overflow-y-auto overscroll-contain mask-[linear-gradient(to_bottom,transparent_0%,rgb(0_0_0/22%)_13%,black_34%,black_100%)] py-[1.55rem] pr-[0.65rem] pb-[0.2rem] pl-0 [-webkit-mask-image:linear-gradient(to_bottom,transparent_0%,rgb(0_0_0/22%)_13%,black_34%,black_100%)] [scrollbar-width:none] max-[640px]:pr-[0.35rem] max-[620px]:landscape:top-5 max-[620px]:landscape:h-[calc(100%-1.25rem)] [&::-webkit-scrollbar]:hidden"
                            aria-live="polite"
                            onScroll={(event) => {
                                const target = event.currentTarget
                                const distanceFromBottom =
                                    target.scrollHeight - target.scrollTop - target.clientHeight
                                shouldAutoScrollTranscriptRef.current = distanceFromBottom < 24
                            }}
                        >
                            <p className="m-0 max-w-full font-heading text-[clamp(1.85rem,4.35vw,4.2rem)] leading-[1.02] font-extrabold tracking-normal [overflow-wrap:anywhere] text-white [text-shadow:0_0.18rem_0.9rem_rgb(0_0_0_/_82%),0_0.85rem_2rem_rgb(0_0_0_/_62%)] max-[640px]:text-[clamp(1.55rem,9vw,3rem)] max-[620px]:landscape:text-[clamp(1.25rem,4vw,2.25rem)]">
                                {sentenceText}
                            </p>
                        </div>
                    </div>

                    <div className="flex flex-wrap items-center gap-[0.55rem] justify-self-start max-[640px]:grid max-[640px]:w-full max-[640px]:grid-cols-5 max-[640px]:gap-[0.42rem] max-[620px]:landscape:justify-self-end">
                        <button
                            className={cn(
                                dockButtonClass,
                                isDetecting
                                    ? "border-[rgb(255_205_77_/_34%)] bg-[rgb(255_205_77_/_82%)] text-[rgb(4_13_23)]"
                                    : "border-[rgb(36_214_190_/_36%)] bg-[rgb(36_214_190_/_84%)] text-[rgb(4_13_23)]"
                            )}
                            onClick={() => void toggleDetection()}
                            type="button"
                        >
                            <HugeiconsIcon
                                icon={isDetecting ? VideoOffIcon : PlayIcon}
                                size={18}
                            />
                            {isDetecting ? "Pause" : "Start"}
                        </button>
                        <button
                            className={cn(
                                dockButtonClass,
                                isSpeaking
                                    ? "border-[rgb(255_205_77_/_34%)] bg-[rgb(255_205_77_/_82%)] text-[rgb(4_13_23)]"
                                    : "border-[rgb(36_214_190_/_28%)]",
                                !isSpeaking &&
                                    (!speechSupported || !sentence.trim()) &&
                                    "cursor-not-allowed opacity-45 hover:translate-y-0"
                            )}
                            disabled={
                                !isSpeaking && (!speechSupported || !sentence.trim())
                            }
                            onClick={speakTranscript}
                            type="button"
                            title={
                                speechSupported
                                    ? "Read transcription aloud"
                                    : "Text-to-speech is unavailable"
                            }
                        >
                            <HugeiconsIcon icon={VolumeHighIcon} size={18} />
                            {isSpeaking ? "Stop" : "Speak"}
                        </button>
                        <button
                            className={dockButtonClass}
                            onClick={() => void sendControl("backspace")}
                            type="button"
                        >
                            <HugeiconsIcon icon={ArrowTurnBackwardIcon} size={18} />
                            Backspace
                        </button>
                        <button
                            className={dockButtonClass}
                            onClick={() => void sendControl("delete_word")}
                            type="button"
                        >
                            <HugeiconsIcon icon={Delete02Icon} size={18} />
                            Word
                        </button>
                        <button
                            className={dockButtonClass}
                            onClick={() => void sendControl("clear")}
                            type="button"
                        >
                            <HugeiconsIcon icon={WasteIcon} size={18} />
                            Clear
                        </button>
                    </div>
                </div>
            </section>
        </main>
    )
}
