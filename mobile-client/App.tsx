import { Ionicons } from "@expo/vector-icons";
import { CameraView, useCameraPermissions } from "expo-camera";
import { LinearGradient } from "expo-linear-gradient";
import * as Speech from "expo-speech";
import { StatusBar } from "expo-status-bar";
import type { ComponentProps, ReactNode } from "react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
    ActivityIndicator,
    Animated,
    Image,
    LayoutChangeEvent,
    Linking,
    Pressable,
    ScrollView,
    StyleSheet,
    Text,
    useWindowDimensions,
    View,
} from "react-native";
import { SafeAreaProvider, useSafeAreaInsets } from "react-native-safe-area-context";

import { SanketAnimatedLogo } from "./components/SanketAnimatedLogo";

type Mode = "static" | "motion";

type LandmarkPoint = {
    x: number;
    y: number;
};

type LandmarkPayloadPoint = LandmarkPoint | [number, number];

type PredictResponse = {
    session_id: string;
    mode: Mode;
    hand_detected: boolean;
    current_prediction: string | null;
    confidence: number | null;
    landmarks: LandmarkPayloadPoint[];
    sentence: string;
    added_to_sentence: boolean;
    buffer_progress: number;
    sequence_progress: number;
};

type ControlAction =
    | "clear"
    | "backspace"
    | "delete_word"
    | "reset_tracking";

type FrameSize = {
    width: number;
    height: number;
};

type IoniconName = ComponentProps<typeof Ionicons>["name"];

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
];

const STATIC_INTERVAL_MS = 260;
const MOTION_INTERVAL_MS = 110;
const PREDICT_TIMEOUT_MS = 8000;
const STATIC_CAPTURE_LONG_EDGE = 640;
const MOTION_CAPTURE_LONG_EDGE = 420;
const MIN_CAPTURE_LONG_EDGE = 360;

const API_BASE =
    process.env.EXPO_PUBLIC_INFERENCE_API_URL?.replace(/\/$/, "") ??
    "http://127.0.0.1:8000";

const signReferenceSource = require("./assets/sign-reference.jpeg");

const HELP_LINKS = [
    {
        label: "Sign Language dictionary",
        href: "https://indiansignlanguage.org/dictionary/",
    },
    {
        label: "Resources",
        href: "https://islrtc.nic.in/",
    },
    {
        label: "Backend health",
        href: `${API_BASE}/health`,
    },
];

function normalizeLandmarkPayload(points: LandmarkPayloadPoint[]): LandmarkPoint[] {
    return points
        .map((point) => {
            if (Array.isArray(point)) {
                const [x, y] = point;
                return typeof x === "number" && typeof y === "number" ? { x, y } : null;
            }

            return typeof point?.x === "number" && typeof point?.y === "number" ? point : null;
        })
        .filter((point): point is LandmarkPoint => point !== null);
}

function createSessionId() {
    return `mobile-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function parsePictureSize(size: string): FrameSize | null {
    const match = /^(\d+)x(\d+)$/.exec(size);
    if (!match) {
        return null;
    }

    const width = Number(match[1]);
    const height = Number(match[2]);
    return width > 0 && height > 0 ? { width, height } : null;
}

function choosePictureSize(sizes: string[], mode: Mode) {
    const targetLongEdge =
        mode === "motion" ? MOTION_CAPTURE_LONG_EDGE : STATIC_CAPTURE_LONG_EDGE;
    const scored = sizes
        .map((size) => {
            const parsed = parsePictureSize(size);
            if (!parsed) {
                return null;
            }

            const longEdge = Math.max(parsed.width, parsed.height);
            const shortEdge = Math.min(parsed.width, parsed.height);
            const aspect = longEdge / shortEdge;
            const aspectPenalty = Math.abs(aspect - 4 / 3) * 500;
            const sizePenalty =
                longEdge > targetLongEdge
                    ? (longEdge - targetLongEdge) * 3
                    : targetLongEdge - longEdge;
            const tooSmallPenalty = longEdge < MIN_CAPTURE_LONG_EDGE ? 1000 : 0;

            return { size, score: aspectPenalty + sizePenalty + tooSmallPenalty };
        })
        .filter((entry): entry is { size: string; score: number } => entry !== null)
        .sort((a, b) => a.score - b.score);

    return scored[0]?.size;
}

function clampPercent(value: number) {
    return Math.max(0, Math.min(100, value));
}

function mapLandmarkToPreview(
    point: LandmarkPoint,
    preview: FrameSize,
    sourceSize: FrameSize | null,
): LandmarkPoint {
    if (!sourceSize?.width || !sourceSize.height) {
        return {
            x: point.x * preview.width,
            y: point.y * preview.height,
        };
    }

    const scale = Math.max(preview.width / sourceSize.width, preview.height / sourceSize.height);
    const renderedWidth = sourceSize.width * scale;
    const renderedHeight = sourceSize.height * scale;
    const offsetX = (preview.width - renderedWidth) / 2;
    const offsetY = (preview.height - renderedHeight) / 2;

    return {
        x: offsetX + point.x * renderedWidth,
        y: offsetY + point.y * renderedHeight,
    };
}

function getLandmarkDotStyle(point: LandmarkPoint, preview: FrameSize, sourceSize: FrameSize | null) {
    const mapped = mapLandmarkToPreview(point, preview, sourceSize);
    return {
        left: mapped.x - 4,
        top: mapped.y - 4,
    };
}

function getConnectionStyle(start: LandmarkPoint, end: LandmarkPoint) {
    const deltaX = end.x - start.x;
    const deltaY = end.y - start.y;
    const length = Math.sqrt(deltaX * deltaX + deltaY * deltaY);
    const angle = `${Math.atan2(deltaY, deltaX)}rad`;

    return {
        left: start.x,
        top: start.y,
        width: length,
        transform: [{ rotate: angle }],
    };
}

function ProgressBar({ value }: { value: number }) {
    return (
        <View style={styles.progressTrack}>
            <LinearGradient
                colors={["#24d6be", "#ffcd4d"]}
                start={{ x: 0, y: 0.5 }}
                end={{ x: 1, y: 0.5 }}
                style={[styles.progressFill, { width: `${clampPercent(value)}%` }]}
            />
        </View>
    );
}

function Pill({
    children,
    tone = "neutral",
}: {
    children: ReactNode;
    tone?: "neutral" | "ok" | "bad" | "live";
}) {
    return (
        <View
            style={[
                styles.pill,
                tone === "ok" && styles.pillOk,
                tone === "bad" && styles.pillBad,
                tone === "live" && styles.pillLive,
            ]}
        >
            <Text style={styles.pillText} numberOfLines={1}>
                {children}
            </Text>
        </View>
    );
}

function Metric({
    label,
    value,
    percent,
}: {
    label: string;
    value: string;
    percent: number;
}) {
    return (
        <View style={styles.metricCard}>
            <Text style={styles.metricLabel}>{label}</Text>
            <Text style={styles.metricValue} numberOfLines={1} adjustsFontSizeToFit>
                {value}
            </Text>
            <ProgressBar value={percent} />
        </View>
    );
}

function DockButton({
    icon,
    label,
    onPress,
    active = false,
    disabled = false,
}: {
    icon: IoniconName;
    label: string;
    onPress: () => void;
    active?: boolean;
    disabled?: boolean;
}) {
    return (
        <Pressable
            accessibilityLabel={label}
            accessibilityRole="button"
            disabled={disabled}
            onPress={onPress}
            style={({ pressed }) => [
                styles.dockButton,
                active && styles.dockButtonActive,
                disabled && styles.dockButtonDisabled,
                pressed && !disabled && styles.dockButtonPressed,
            ]}
        >
            <Ionicons
                name={icon}
                color={active ? "#04111b" : "#e7f5fb"}
                size={18}
            />
            <Text
                style={[
                    styles.dockButtonText,
                    active && styles.dockButtonTextActive,
                    disabled && styles.dockButtonTextDisabled,
                ]}
                numberOfLines={1}
                adjustsFontSizeToFit
            >
                {label}
            </Text>
        </Pressable>
    );
}

function SanketApp() {
    const cameraRef = useRef<CameraView | null>(null);
    const inFlightRef = useRef(false);
    const helpDrawerProgress = useRef(new Animated.Value(0)).current;
    const { width: windowWidth, height: windowHeight } = useWindowDimensions();
    const insets = useSafeAreaInsets();
    const [permission, requestPermission] = useCameraPermissions();
    const [cameraReady, setCameraReady] = useState(false);
    const [pictureSizes, setPictureSizes] = useState<string[]>([]);
    const [previewLayout, setPreviewLayout] = useState({ width: 0, height: 0 });
    const [sourceFrameSize, setSourceFrameSize] = useState<FrameSize | null>(null);

    const [sessionId, setSessionId] = useState("");
    const [mode, setMode] = useState<Mode>("static");
    const [isDetecting, setIsDetecting] = useState(false);
    const [isSending, setIsSending] = useState(false);
    const [isSpeaking, setIsSpeaking] = useState(false);
    const [isHelpOpen, setIsHelpOpen] = useState(false);
    const [isIntroVisible, setIsIntroVisible] = useState(true);
    const [serverOnline, setServerOnline] = useState<boolean | null>(null);

    const [handDetected, setHandDetected] = useState(false);
    const [currentPrediction, setCurrentPrediction] = useState("-");
    const [confidence, setConfidence] = useState<number | null>(null);
    const [sentence, setSentence] = useState("");
    const [bufferProgress, setBufferProgress] = useState(0);
    const [sequenceProgress, setSequenceProgress] = useState(0);
    const [landmarks, setLandmarks] = useState<LandmarkPoint[]>([]);
    const [errorMessage, setErrorMessage] = useState<string | null>(null);

    useEffect(() => {
        setSessionId(createSessionId());
    }, []);

    useEffect(() => {
        const timer = setTimeout(() => {
            setIsIntroVisible(false);
        }, 2200);

        return () => {
            clearTimeout(timer);
        };
    }, []);

    const pictureSize = useMemo(
        () => choosePictureSize(pictureSizes, mode),
        [mode, pictureSizes],
    );

    const resetLocalTracking = useCallback(() => {
        setHandDetected(false);
        setCurrentPrediction("-");
        setConfidence(null);
        setBufferProgress(0);
        setSequenceProgress(0);
        setLandmarks([]);
        setSourceFrameSize(null);
    }, []);

    const handleCameraReady = useCallback(() => {
        setCameraReady(true);

        void cameraRef.current
            ?.getAvailablePictureSizesAsync()
            .then((sizes) => {
                setPictureSizes(sizes);
            })
            .catch(() => {
                setPictureSizes([]);
            });
    }, []);

    const checkHealth = useCallback(async () => {
        try {
            const response = await fetch(`${API_BASE}/health`);
            if (!response.ok) {
                throw new Error(`Health check failed with status ${response.status}`);
            }
            setServerOnline(true);
        } catch {
            setServerOnline(false);
        }
    }, []);

    useEffect(() => {
        void checkHealth();

        const timer = setInterval(() => {
            void checkHealth();
        }, 10000);

        return () => {
            clearInterval(timer);
        };
    }, [checkHealth]);

    const sendControl = useCallback(
        async (action: ControlAction) => {
            if (!sessionId) {
                return;
            }

            try {
                const response = await fetch(`${API_BASE}/control`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ session_id: sessionId, action }),
                });

                if (!response.ok) {
                    throw new Error(`Control request failed with status ${response.status}`);
                }

                const data = (await response.json()) as { sentence: string };
                setSentence(data.sentence);
                setErrorMessage(null);
            } catch {
                setErrorMessage("Failed to send control action to backend.");
            }
        },
        [sessionId],
    );

    const changeMode = useCallback(
        (nextMode: Mode) => {
            setMode(nextMode);
            resetLocalTracking();
            void sendControl("reset_tracking");
        },
        [resetLocalTracking, sendControl],
    );

    const sendPrediction = useCallback(async () => {
        if (!sessionId || !cameraReady || !isDetecting || !cameraRef.current) {
            return;
        }

        setIsSending(true);
        setErrorMessage(null);

        try {
            const photo = await cameraRef.current.takePictureAsync({
                base64: true,
                quality: mode === "motion" ? 0.24 : 0.34,
                shutterSound: false,
            });

            if (!photo?.base64) {
                throw new Error("Camera did not return a base64 frame.");
            }

            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), PREDICT_TIMEOUT_MS);
            const response = await fetch(`${API_BASE}/predict`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    session_id: sessionId,
                    mode,
                    image_data: `data:image/jpeg;base64,${photo.base64}`,
                    flip_horizontal: false,
                }),
                signal: controller.signal,
            }).finally(() => {
                clearTimeout(timeoutId);
            });

            if (!response.ok) {
                throw new Error(`Predict request failed with status ${response.status}`);
            }

            const data = (await response.json()) as PredictResponse;
            setServerOnline(true);
            setHandDetected(data.hand_detected);
            setCurrentPrediction(data.current_prediction ?? "-");
            setConfidence(data.confidence);
            setSentence(data.sentence);
            setBufferProgress(data.buffer_progress);
            setSequenceProgress(data.sequence_progress);
            setSourceFrameSize({ width: photo.width, height: photo.height });
            setLandmarks(normalizeLandmarkPayload(data.landmarks ?? []));
        } catch {
            setServerOnline(false);
            setErrorMessage("Prediction failed. Check the backend URL and ngrok tunnel.");
        } finally {
            setIsSending(false);
        }
    }, [cameraReady, isDetecting, mode, sessionId]);

    useEffect(() => {
        if (!isDetecting || permission?.granted !== true) {
            return;
        }

        const timer = setInterval(() => {
            if (inFlightRef.current) {
                return;
            }

            inFlightRef.current = true;
            void sendPrediction().finally(() => {
                inFlightRef.current = false;
            });
        }, mode === "motion" ? MOTION_INTERVAL_MS : STATIC_INTERVAL_MS);

        return () => {
            clearInterval(timer);
            inFlightRef.current = false;
        };
    }, [isDetecting, mode, permission?.granted, sendPrediction]);

    const startDetection = useCallback(async () => {
        setErrorMessage(null);

        if (permission?.granted !== true) {
            const result = await requestPermission();
            if (!result.granted) {
                setErrorMessage("Camera access is needed to open the live view.");
                return;
            }
        }

        resetLocalTracking();
        await sendControl("reset_tracking");
        setIsDetecting(true);
    }, [permission?.granted, requestPermission, resetLocalTracking, sendControl]);

    const toggleDetection = useCallback(async () => {
        if (isDetecting) {
            setIsDetecting(false);
            return;
        }

        await startDetection();
    }, [isDetecting, startDetection]);

    const speakTranscript = useCallback(async () => {
        const text = sentence.trim();

        if (await Speech.isSpeakingAsync()) {
            Speech.stop();
            setIsSpeaking(false);
            return;
        }

        if (!text) {
            setErrorMessage("There is no transcription to read yet.");
            return;
        }

        setErrorMessage(null);
        setIsSpeaking(true);
        Speech.speak(text, {
            language: "en",
            pitch: 1,
            rate: 0.92,
            onDone: () => setIsSpeaking(false),
            onStopped: () => setIsSpeaking(false),
            onError: () => {
                setIsSpeaking(false);
                setErrorMessage("Could not play the transcription voice.");
            },
        });
    }, [sentence]);

    useEffect(() => {
        return () => {
            Speech.stop();
        };
    }, []);

    useEffect(() => {
        Animated.timing(helpDrawerProgress, {
            toValue: isHelpOpen ? 1 : 0,
            duration: 260,
            useNativeDriver: true,
        }).start();
    }, [helpDrawerProgress, isHelpOpen]);

    const onPreviewLayout = (event: LayoutChangeEvent) => {
        const { width, height } = event.nativeEvent.layout;
        setPreviewLayout({ width, height });
    };

    const openResource = useCallback((href: string) => {
        void Linking.openURL(href);
    }, []);

    const hasCameraAccess = permission?.granted === true;
    const confidenceValue = confidence === null ? 0 : clampPercent(confidence * 100);
    const confidenceText = confidence === null ? "-" : `${confidenceValue.toFixed(1)}%`;
    const progressValue = mode === "static" ? bufferProgress : sequenceProgress;
    const progressTotal = mode === "static" ? 10 : 30;
    const progressPercent = (progressValue / progressTotal) * 100;
    const sentenceText = sentence.trim() || (isDetecting ? "Listening..." : "Ready");
    const apiTone = serverOnline === null ? "neutral" : serverOnline ? "ok" : "bad";
    const previewHeight = Math.max(520, windowHeight);
    const compact = windowWidth < 380;
    const topOffset = Math.max(insets.top, 10);
    const bottomOffset = Math.max(insets.bottom, 10);
    const helpDrawerWidth = Math.min(360, Math.max(300, windowWidth * 0.88));
    const helpDrawerTranslateX = helpDrawerProgress.interpolate({
        inputRange: [0, 1],
        outputRange: [helpDrawerWidth, 0],
    });

    return (
        <View style={styles.safeArea}>
            <StatusBar style="light" />
            <View style={styles.root}>
                <View style={styles.cameraStage} onLayout={onPreviewLayout}>
                    {hasCameraAccess ? (
                        <CameraView
                            ref={cameraRef}
                            active
                            animateShutter={false}
                            facing="front"
                            mirror
                            onCameraReady={handleCameraReady}
                            pictureSize={pictureSize}
                            style={[styles.camera, { minHeight: previewHeight }]}
                        />
                    ) : (
                        <View style={styles.cameraPlaceholder}>
                            <Ionicons name="videocam-outline" color="#24d6be" size={46} />
                            <Text style={styles.cameraPlaceholderText}>
                                {hasCameraAccess
                                    ? "Camera is ready."
                                    : "Camera permission is required."}
                            </Text>
                            {!hasCameraAccess && (
                                <Pressable
                                    accessibilityRole="button"
                                    onPress={() => void requestPermission()}
                                    style={({ pressed }) => [
                                        styles.permissionButton,
                                        pressed && styles.dockButtonPressed,
                                    ]}
                                >
                                    <Ionicons name="camera-outline" color="#04111b" size={18} />
                                    <Text style={styles.permissionButtonText}>Enable camera</Text>
                                </Pressable>
                            )}
                        </View>
                    )}

                    {previewLayout.width > 0 && previewLayout.height > 0 && (
                        <View pointerEvents="none" style={styles.landmarkLayer}>
                            {HAND_CONNECTIONS.map(([startIdx, endIdx]) => {
                                const start = landmarks[startIdx];
                                const end = landmarks[endIdx];
                                if (!start || !end) {
                                    return null;
                                }

                                const mappedStart = mapLandmarkToPreview(
                                    start,
                                    previewLayout,
                                    sourceFrameSize,
                                );
                                const mappedEnd = mapLandmarkToPreview(
                                    end,
                                    previewLayout,
                                    sourceFrameSize,
                                );

                                return (
                                    <View
                                        key={`${startIdx}-${endIdx}`}
                                        style={[
                                            styles.landmarkLine,
                                            getConnectionStyle(mappedStart, mappedEnd),
                                        ]}
                                    />
                                );
                            })}
                            {landmarks.map((point, index) => (
                                <View
                                    key={index}
                                    style={[
                                        styles.landmarkDot,
                                        getLandmarkDotStyle(point, previewLayout, sourceFrameSize),
                                    ]}
                                />
                            ))}
                        </View>
                    )}

                    <LinearGradient
                        pointerEvents="none"
                        colors={[
                            "rgba(0,0,0,0.42)",
                            "rgba(0,0,0,0.02)",
                            "rgba(0,0,0,0.26)",
                        ]}
                        locations={[0, 0.42, 1]}
                        start={{ x: 0, y: 0 }}
                        end={{ x: 1, y: 0 }}
                        style={styles.sideShade}
                    />
                    <LinearGradient
                        pointerEvents="none"
                        colors={[
                            "rgba(1,5,12,0)",
                            "rgba(1,5,12,0.2)",
                            "rgba(1,5,12,0.74)",
                            "rgba(1,5,12,0.96)",
                        ]}
                        locations={[0, 0.33, 0.72, 1]}
                        style={styles.bottomShade}
                    />
                    {isDetecting && handDetected && (
                        <LinearGradient
                            pointerEvents="none"
                            colors={[
                                "rgba(36,214,190,0.16)",
                                "rgba(36,214,190,0.03)",
                                "rgba(255,205,77,0.08)",
                            ]}
                            start={{ x: 0.5, y: 0 }}
                            end={{ x: 0.5, y: 1 }}
                            style={styles.liveWash}
                        />
                    )}
                </View>

                <View style={[styles.topBar, { top: topOffset + 2 }]}>
                    <View style={styles.logoPlate}>
                        <SanketAnimatedLogo width={118} height={50} />
                    </View>

                    <View style={styles.topControls}>
                        <View style={styles.segment} accessibilityLabel="Recognition mode">
                            <Pressable
                                accessibilityRole="button"
                                onPress={() => changeMode("static")}
                                style={[
                                    styles.segmentButton,
                                    mode === "static" && styles.segmentButtonActive,
                                ]}
                            >
                                <Text
                                    style={[
                                        styles.segmentText,
                                        mode === "static" && styles.segmentTextActive,
                                    ]}
                                >
                                    Static
                                </Text>
                            </Pressable>
                            <Pressable
                                accessibilityRole="button"
                                onPress={() => changeMode("motion")}
                                style={[
                                    styles.segmentButton,
                                    mode === "motion" && styles.segmentButtonActive,
                                ]}
                            >
                                <Text
                                    style={[
                                        styles.segmentText,
                                        mode === "motion" && styles.segmentTextActive,
                                    ]}
                                >
                                    Motion
                                </Text>
                            </Pressable>
                        </View>
                        <Pressable
                            accessibilityLabel="Open help book"
                            accessibilityRole="button"
                            onPress={() => setIsHelpOpen(true)}
                            style={({ pressed }) => [
                                styles.helpButton,
                                pressed && styles.dockButtonPressed,
                            ]}
                        >
                            <Ionicons name="help-circle-outline" color="#e7fffb" size={23} />
                        </Pressable>
                    </View>
                </View>

                <View style={[styles.statusRow, { top: topOffset + 62 }]}>
                    <Pill tone={apiTone}>
                        {serverOnline === null
                            ? "Checking server"
                            : serverOnline
                                ? "Server live"
                                : "Server offline"}
                    </Pill>
                    <Pill tone={isDetecting ? "live" : "neutral"}>
                        {isDetecting ? (handDetected ? "Hand locked" : "Seeking hand") : "Paused"}
                    </Pill>
                    {!compact && <Pill>{isSending ? "Frame sending" : "Frame idle"}</Pill>}
                </View>

                <View style={[styles.metricsRail, { top: topOffset + 106 }]}>
                    <Metric label="Confidence" value={confidenceText} percent={confidenceValue} />
                    <Metric
                        label="Progress"
                        value={`${progressValue}/${progressTotal}`}
                        percent={progressPercent}
                    />
                </View>

                <Animated.View
                    pointerEvents={isHelpOpen ? "auto" : "none"}
                    style={[styles.helpScrim, { opacity: helpDrawerProgress }]}
                >
                    <Pressable
                        accessibilityLabel="Close help book"
                        accessibilityRole="button"
                        onPress={() => setIsHelpOpen(false)}
                        style={StyleSheet.absoluteFill}
                    />
                </Animated.View>

                <Animated.View
                    pointerEvents={isHelpOpen ? "auto" : "none"}
                    accessibilityViewIsModal={isHelpOpen}
                    style={[
                        styles.helpSheet,
                        {
                            width: helpDrawerWidth,
                            paddingTop: topOffset + 10,
                            paddingBottom: bottomOffset + 14,
                            transform: [{ translateX: helpDrawerTranslateX }],
                        },
                    ]}
                >
                    <View style={styles.helpHeader}>
                        <View style={styles.helpTitleGroup}>
                            <View style={styles.helpHeaderIcon}>
                                <Ionicons name="book-outline" color="#24d6be" size={22} />
                            </View>
                            <View style={styles.helpTitleCopy}>
                                <Text style={styles.helpEyebrow}>Quick reference</Text>
                                <Text style={styles.helpTitle}>Help book</Text>
                            </View>
                        </View>
                        <Pressable
                            accessibilityLabel="Close help book"
                            accessibilityRole="button"
                            onPress={() => setIsHelpOpen(false)}
                            style={({ pressed }) => [
                                styles.helpCloseButton,
                                pressed && styles.dockButtonPressed,
                            ]}
                        >
                            <Ionicons name="close" color="#ffffff" size={22} />
                        </Pressable>
                    </View>

                    <ScrollView
                        contentContainerStyle={styles.helpScroll}
                        showsVerticalScrollIndicator={false}
                    >
                        <View style={styles.helpImageFrame}>
                            <Image
                                source={signReferenceSource}
                                resizeMode="contain"
                                style={styles.helpImage}
                            />
                        </View>

                        <View style={styles.helpLinks}>
                            {HELP_LINKS.map((link) => (
                                <Pressable
                                    accessibilityRole="link"
                                    key={link.href}
                                    onPress={() => openResource(link.href)}
                                    style={({ pressed }) => [
                                        styles.helpLink,
                                        pressed && styles.dockButtonPressed,
                                    ]}
                                >
                                    <Text
                                        style={styles.helpLinkText}
                                        numberOfLines={1}
                                        adjustsFontSizeToFit
                                    >
                                        {link.label}
                                    </Text>
                                    <Ionicons name="open-outline" color="#e7fffb" size={18} />
                                </Pressable>
                            ))}
                        </View>
                    </ScrollView>
                </Animated.View>

                {!isDetecting && hasCameraAccess && (
                    <View style={styles.startPrompt}>
                        <Pressable
                            accessibilityLabel="Start sign detection"
                            accessibilityRole="button"
                            onPress={() => void startDetection()}
                            style={({ pressed }) => [
                                styles.startButton,
                                pressed && styles.dockButtonPressed,
                            ]}
                        >
                            <Ionicons name="play" color="#05111b" size={32} />
                        </Pressable>
                        <View style={styles.startCopy}>
                            <Text style={styles.startEyebrow}>Sanket is ready</Text>
                            <Text style={styles.startTitle}>Start sign detection</Text>
                        </View>
                    </View>
                )}

                <View style={[styles.bottomPanel, { bottom: bottomOffset + 10 }]}>
                    {errorMessage && (
                        <View style={styles.errorPill}>
                            <Text style={styles.errorText}>{errorMessage}</Text>
                        </View>
                    )}

                    <View style={styles.transcriptBlock}>
                        <ScrollView
                            contentContainerStyle={styles.transcriptScroll}
                            showsVerticalScrollIndicator={false}
                        >
                            <Text style={styles.transcriptText}>{sentenceText}</Text>
                        </ScrollView>
                    </View>

                    <View style={styles.currentRow}>
                        <Text style={styles.currentText} numberOfLines={1}>
                            {currentPrediction !== "-" ? currentPrediction : "No sign yet"}
                        </Text>
                        <Text style={styles.apiText} numberOfLines={1}>
                            {API_BASE.replace(/^https?:\/\//, "")}
                        </Text>
                        {isSending && <ActivityIndicator color="#24d6be" size="small" />}
                    </View>

                    <View style={styles.dock}>
                        <DockButton
                            active={isDetecting}
                            icon={isDetecting ? "pause" : "play"}
                            label={isDetecting ? "Pause" : "Start"}
                            onPress={() => void toggleDetection()}
                        />
                        <DockButton
                            active={isSpeaking}
                            disabled={!isSpeaking && !sentence.trim()}
                            icon={isSpeaking ? "stop" : "volume-high"}
                            label={isSpeaking ? "Stop" : "Speak"}
                            onPress={() => void speakTranscript()}
                        />
                        <DockButton
                            icon="backspace-outline"
                            label="Back"
                            onPress={() => void sendControl("backspace")}
                        />
                        <DockButton
                            icon="text-outline"
                            label="Word"
                            onPress={() => void sendControl("delete_word")}
                        />
                        <DockButton
                            icon="trash-outline"
                            label="Clear"
                            onPress={() => void sendControl("clear")}
                        />
                    </View>
                </View>

                {isIntroVisible && (
                    <View pointerEvents="none" style={styles.introOverlay}>
                        <SanketAnimatedLogo width={230} height={230} />
                    </View>
                )}
            </View>
        </View>
    );
}

export default function App() {
    return (
        <SafeAreaProvider>
            <SanketApp />
        </SafeAreaProvider>
    );
}

const glassSurface = {
    borderWidth: 1,
    borderColor: "rgba(255, 255, 255, 0.16)",
    backgroundColor: "rgba(5, 12, 23, 0.58)",
    shadowColor: "#000000",
    shadowOpacity: 0.24,
    shadowRadius: 24,
    shadowOffset: { width: 0, height: 14 },
    elevation: 6,
};

const styles = StyleSheet.create({
    safeArea: {
        flex: 1,
        backgroundColor: "#04111b",
    },
    root: {
        flex: 1,
        overflow: "hidden",
        backgroundColor: "#04111b",
    },
    cameraStage: {
        ...StyleSheet.absoluteFillObject,
        backgroundColor: "#020711",
    },
    camera: {
        ...StyleSheet.absoluteFillObject,
        width: "100%",
    },
    cameraPlaceholder: {
        ...StyleSheet.absoluteFillObject,
        alignItems: "center",
        justifyContent: "center",
        gap: 14,
        padding: 28,
        backgroundColor: "#020711",
    },
    cameraPlaceholderText: {
        maxWidth: 300,
        color: "#e6eef6",
        fontSize: 16,
        fontWeight: "800",
        lineHeight: 22,
        textAlign: "center",
    },
    permissionButton: {
        alignItems: "center",
        flexDirection: "row",
        gap: 8,
        minHeight: 42,
        borderRadius: 999,
        backgroundColor: "#24d6be",
        paddingHorizontal: 16,
    },
    permissionButtonText: {
        color: "#04111b",
        fontSize: 14,
        fontWeight: "900",
    },
    landmarkLayer: {
        ...StyleSheet.absoluteFillObject,
        zIndex: 2,
    },
    landmarkLine: {
        position: "absolute",
        height: 3,
        borderRadius: 999,
        backgroundColor: "rgba(36, 214, 190, 0.94)",
        transformOrigin: "left center",
    },
    landmarkDot: {
        position: "absolute",
        width: 8,
        height: 8,
        borderRadius: 4,
        backgroundColor: "rgba(255, 205, 77, 0.96)",
    },
    sideShade: {
        ...StyleSheet.absoluteFillObject,
    },
    bottomShade: {
        position: "absolute",
        right: 0,
        bottom: 0,
        left: 0,
        height: "66%",
    },
    liveWash: {
        ...StyleSheet.absoluteFillObject,
        borderWidth: 1,
        borderColor: "rgba(36, 214, 190, 0.26)",
    },
    topBar: {
        position: "absolute",
        top: 12,
        left: 14,
        right: 14,
        zIndex: 5,
        alignItems: "center",
        flexDirection: "row",
        justifyContent: "space-between",
        gap: 12,
    },
    logoPlate: {
        ...glassSurface,
        width: 128,
        height: 54,
        alignItems: "center",
        justifyContent: "center",
        overflow: "hidden",
        borderRadius: 8,
        backgroundColor: "rgba(5, 12, 23, 0.34)",
    },
    topControls: {
        flexShrink: 1,
        alignItems: "flex-end",
        flexDirection: "row",
        gap: 8,
    },
    segment: {
        ...glassSurface,
        width: 158,
        height: 40,
        flexDirection: "row",
        gap: 3,
        borderRadius: 999,
        padding: 3,
    },
    segmentButton: {
        flex: 1,
        alignItems: "center",
        justifyContent: "center",
        borderRadius: 999,
    },
    segmentButtonActive: {
        backgroundColor: "#f6fafc",
    },
    segmentText: {
        color: "#d2e1ee",
        fontSize: 13,
        fontWeight: "900",
    },
    segmentTextActive: {
        color: "#04111b",
    },
    helpButton: {
        ...glassSurface,
        width: 40,
        height: 40,
        alignItems: "center",
        justifyContent: "center",
        borderRadius: 999,
        borderColor: "rgba(36, 214, 190, 0.28)",
        backgroundColor: "rgba(6, 20, 32, 0.62)",
    },
    statusRow: {
        position: "absolute",
        top: 72,
        left: 14,
        right: 14,
        zIndex: 5,
        flexDirection: "row",
        flexWrap: "wrap",
        gap: 7,
    },
    pill: {
        ...glassSurface,
        minHeight: 32,
        justifyContent: "center",
        borderRadius: 999,
        paddingHorizontal: 10,
    },
    pillOk: {
        borderColor: "rgba(44, 214, 190, 0.38)",
        backgroundColor: "rgba(18, 141, 123, 0.34)",
    },
    pillBad: {
        borderColor: "rgba(248, 92, 92, 0.42)",
        backgroundColor: "rgba(137, 24, 42, 0.42)",
    },
    pillLive: {
        borderColor: "rgba(36, 214, 190, 0.5)",
        backgroundColor: "rgba(8, 60, 58, 0.5)",
    },
    pillText: {
        color: "#e2ecf6",
        fontSize: 12,
        fontWeight: "900",
    },
    metricsRail: {
        position: "absolute",
        top: 116,
        right: 14,
        zIndex: 5,
        width: 154,
        gap: 8,
    },
    metricCard: {
        ...glassSurface,
        gap: 7,
        borderRadius: 8,
        padding: 10,
    },
    metricLabel: {
        color: "#b1c5d8",
        fontSize: 10,
        fontWeight: "900",
        textTransform: "uppercase",
    },
    metricValue: {
        color: "#ffffff",
        fontSize: 18,
        fontWeight: "900",
    },
    progressTrack: {
        height: 5,
        overflow: "hidden",
        borderRadius: 999,
        backgroundColor: "rgba(255, 255, 255, 0.16)",
    },
    progressFill: {
        height: "100%",
        borderRadius: 999,
        backgroundColor: "#24d6be",
    },
    startPrompt: {
        ...glassSurface,
        position: "absolute",
        top: "38%",
        left: 18,
        right: 18,
        zIndex: 5,
        alignSelf: "center",
        alignItems: "center",
        flexDirection: "row",
        gap: 14,
        borderRadius: 18,
        padding: 12,
        backgroundColor: "rgba(4, 12, 22, 0.68)",
    },
    startButton: {
        width: 62,
        height: 62,
        alignItems: "center",
        justifyContent: "center",
        borderRadius: 31,
        backgroundColor: "#24d6be",
    },
    startCopy: {
        flex: 1,
        gap: 3,
    },
    startEyebrow: {
        color: "#b9d2e2",
        fontSize: 12,
        fontWeight: "900",
        textTransform: "uppercase",
    },
    startTitle: {
        color: "#ffffff",
        fontSize: 22,
        fontWeight: "900",
        lineHeight: 26,
    },
    helpScrim: {
        ...StyleSheet.absoluteFillObject,
        zIndex: 8,
        backgroundColor: "rgba(1, 5, 12, 0.58)",
    },
    helpSheet: {
        position: "absolute",
        top: 0,
        right: 0,
        bottom: 0,
        zIndex: 9,
        gap: 16,
        borderLeftWidth: 1,
        borderLeftColor: "rgba(255, 255, 255, 0.14)",
        backgroundColor: "rgba(4, 13, 23, 0.96)",
        paddingHorizontal: 16,
        shadowColor: "#000000",
        shadowOpacity: 0.36,
        shadowRadius: 28,
        shadowOffset: { width: -10, height: 0 },
        elevation: 12,
    },
    helpHeader: {
        alignItems: "center",
        flexDirection: "row",
        justifyContent: "space-between",
        gap: 12,
    },
    helpTitleGroup: {
        flex: 1,
        alignItems: "center",
        flexDirection: "row",
        gap: 10,
    },
    helpHeaderIcon: {
        width: 42,
        height: 42,
        alignItems: "center",
        justifyContent: "center",
        borderRadius: 999,
        borderWidth: 1,
        borderColor: "rgba(36, 214, 190, 0.3)",
        backgroundColor: "rgba(36, 214, 190, 0.1)",
    },
    helpTitleCopy: {
        flex: 1,
        gap: 2,
    },
    helpEyebrow: {
        color: "#b1c5d8",
        fontSize: 11,
        fontWeight: "900",
        textTransform: "uppercase",
    },
    helpTitle: {
        color: "#ffffff",
        fontSize: 22,
        fontWeight: "900",
        lineHeight: 26,
    },
    helpCloseButton: {
        width: 38,
        height: 38,
        alignItems: "center",
        justifyContent: "center",
        borderRadius: 999,
        borderWidth: 1,
        borderColor: "rgba(255, 255, 255, 0.14)",
        backgroundColor: "rgba(255, 255, 255, 0.08)",
    },
    helpScroll: {
        gap: 14,
        paddingBottom: 6,
    },
    helpImageFrame: {
        minHeight: 330,
        overflow: "hidden",
        borderRadius: 8,
        borderWidth: 1,
        borderColor: "rgba(255, 255, 255, 0.14)",
        backgroundColor: "rgba(255, 255, 255, 0.06)",
    },
    helpImage: {
        width: "100%",
        height: 330,
    },
    helpLinks: {
        gap: 8,
    },
    helpLink: {
        ...glassSurface,
        minHeight: 46,
        alignItems: "center",
        flexDirection: "row",
        justifyContent: "space-between",
        gap: 12,
        borderRadius: 8,
        paddingHorizontal: 12,
        backgroundColor: "rgba(6, 20, 32, 0.72)",
    },
    helpLinkText: {
        flex: 1,
        color: "#e7fffb",
        fontSize: 14,
        fontWeight: "900",
    },
    introOverlay: {
        ...StyleSheet.absoluteFillObject,
        zIndex: 20,
        alignItems: "center",
        justifyContent: "center",
        backgroundColor: "#040d17",
    },
    bottomPanel: {
        position: "absolute",
        left: 14,
        right: 14,
        bottom: 12,
        zIndex: 5,
        gap: 10,
    },
    errorPill: {
        ...glassSurface,
        alignSelf: "flex-start",
        borderRadius: 999,
        borderColor: "rgba(248, 92, 92, 0.38)",
        backgroundColor: "rgba(137, 24, 42, 0.48)",
        paddingHorizontal: 12,
        paddingVertical: 8,
    },
    errorText: {
        color: "#ffe0e3",
        fontSize: 12,
        fontWeight: "800",
    },
    transcriptBlock: {
        height: 126,
        overflow: "hidden",
    },
    transcriptScroll: {
        flexGrow: 1,
        justifyContent: "flex-end",
        paddingTop: 18,
        paddingBottom: 4,
    },
    transcriptText: {
        color: "#ffffff",
        fontSize: 34,
        fontWeight: "900",
        lineHeight: 38,
        textShadowColor: "rgba(0, 0, 0, 0.82)",
        textShadowOffset: { width: 0, height: 2 },
        textShadowRadius: 12,
    },
    currentRow: {
        alignItems: "center",
        flexDirection: "row",
        gap: 9,
    },
    currentText: {
        ...glassSurface,
        flexShrink: 1,
        overflow: "hidden",
        borderRadius: 999,
        color: "#dffefa",
        fontSize: 12,
        fontWeight: "900",
        paddingHorizontal: 10,
        paddingVertical: 7,
    },
    apiText: {
        flex: 1,
        color: "#b1c5d8",
        fontSize: 11,
        fontWeight: "700",
        textAlign: "right",
    },
    dock: {
        ...glassSurface,
        flexDirection: "row",
        gap: 6,
        borderRadius: 18,
        padding: 6,
        backgroundColor: "rgba(5, 12, 23, 0.72)",
    },
    dockButton: {
        flex: 1,
        minWidth: 0,
        minHeight: 54,
        alignItems: "center",
        justifyContent: "center",
        gap: 4,
        borderWidth: 1,
        borderColor: "rgba(36, 214, 190, 0.28)",
        borderRadius: 14,
        backgroundColor: "rgba(6, 20, 32, 0.58)",
        paddingHorizontal: 3,
    },
    dockButtonActive: {
        borderColor: "rgba(255, 205, 77, 0.42)",
        backgroundColor: "rgba(255, 205, 77, 0.86)",
    },
    dockButtonDisabled: {
        opacity: 0.45,
    },
    dockButtonPressed: {
        opacity: 0.76,
        transform: [{ scale: 0.98 }],
    },
    dockButtonText: {
        color: "#e7f5fb",
        fontSize: 10,
        fontWeight: "900",
    },
    dockButtonTextActive: {
        color: "#04111b",
    },
    dockButtonTextDisabled: {
        color: "#b1c5d8",
    },
});
