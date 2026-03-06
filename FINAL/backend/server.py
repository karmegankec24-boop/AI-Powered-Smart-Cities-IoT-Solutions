"""
backend/server.py
FastAPI server — wraps camera.py logic without modifying it.
Streams MJPEG video, exposes status/mode/signal control endpoints.
"""

import sys
import os
import cv2
import time
import threading
import numpy as np
from collections import deque
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Add parent directory to path so we can import camera.py functions ──────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from camera import (
    CONFIDENCE_THRESHOLD,
    DETECTION_GAP_MAX,
    MIN_DETECTIONS,
    EMERGENCY_GREEN_SECS,
    VOTE_WINDOW,
    VOTE_MIN,
    NORMAL_CYCLE,
    EMERGENCY_CLASSES,
    find_esp32_port,
    run_yolo,
    draw_overlay,
    get_current_cycle_colour,
    send_signal,
)

from ultralytics import YOLO
import serial

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(title="Camera Vision API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared state (protected by a lock) ─────────────────────────────────────
lock = threading.Lock()

state = {
    # Detection
    "label":        "Scanning...",
    "confidence":   0.0,
    "is_emergency": False,

    # Traffic signal
    "mode":           "NORMAL",      # NORMAL | CONFIRMING | WAITING_CYCLE | EMERGENCY
    "display_colour": "RED",
    "time_remaining": 0.0,
    "hold_progress":  0.0,

    # App mode
    "app_mode":       "AUTO",        # AUTO | MANUAL
    "manual_signal":  "RED",         # active signal in MANUAL mode

    # Latest JPEG bytes for streaming
    "frame_bytes":  None,
}

# ── Latest frame for the MJPEG stream ─────────────────────────────────────
frame_lock  = threading.Lock()
latest_jpeg = None          # bytes

# ── ESP32 serial handle ────────────────────────────────────────────────────
esp = None


def init_esp():
    global esp
    port = find_esp32_port()
    try:
        esp = serial.Serial(port, 115200, timeout=1)
        time.sleep(2)
        print(f"[server] Connected to ESP32 on {port}")
    except Exception as e:
        print(f"[server] WARNING: Could not open ESP32 serial port: {e}")
        esp = None


# ── Background camera + detection thread ───────────────────────────────────
def camera_thread():
    global latest_jpeg, esp

    print("[server] Loading YOLO model …")
    _FINETUNED = os.path.join(os.path.dirname(__file__), "..", "best_model.pt")
    _COCO      = "yolov8n.pt"
    model_path = _FINETUNED if os.path.exists(_FINETUNED) else _COCO
    model = YOLO(model_path)
    print(f"[server] YOLO loaded from {model_path}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[server] ERROR: cannot open webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Detection state
    det_mode        = "NORMAL"
    det_event_times = []
    prev_is_emerg   = False
    emergency_start = None
    cycle_start     = time.time()
    detection_window = deque(maxlen=VOTE_WINDOW)

    if esp:
        send_signal(esp, "RED", force=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        now = time.time()

        with lock:
            current_app_mode = state["app_mode"]
            manual_signal    = state["manual_signal"]

        # ── MANUAL MODE: just run YOLO for visual boxes, override signal ─────
        if current_app_mode == "MANUAL":
            raw_label, raw_conf, _ = run_yolo(model, frame)

            # Build a simple overlay for manual mode
            h, w = frame.shape[:2]
            banner = frame.copy()
            cv2.rectangle(banner, (0, 0), (w, 60), (0, 0, 0), -1)
            cv2.addWeighted(banner, 0.6, frame, 0.4, 0, frame)
            cv2.putText(frame, f"MANUAL MODE  |  Signal: {manual_signal}",
                        (18, 35), cv2.FONT_HERSHEY_DUPLEX, 0.85, (200, 200, 200), 2, cv2.LINE_AA)

            SIGNAL_COLORS_CV = {
                "RED":    (60,  60, 220),
                "GREEN":  (0,  200,  60),
                "YELLOW": (0,  200, 220),
            }
            border_col = SIGNAL_COLORS_CV.get(manual_signal, (160, 160, 160))
            cv2.rectangle(frame, (0, 0), (w-1, h-1), border_col, 5)

            if esp:
                send_signal(esp, manual_signal)

            with lock:
                state.update({
                    "label":          raw_label,
                    "confidence":     raw_conf,
                    "display_colour": manual_signal,
                    "time_remaining": 0.0,
                    "hold_progress":  0.0,
                    "mode":           "MANUAL",
                })

            # Reset auto-mode state so resuming AUTO starts clean
            det_mode        = "NORMAL"
            det_event_times = []
            prev_is_emerg   = False
            emergency_start = None
            cycle_start     = time.time()
            detection_window.clear()

        else:
            # ── AUTO MODE: full camera.py logic ─────────────────────────────
            raw_label, raw_conf, raw_is_emerg = run_yolo(model, frame)

            detection_window.append((raw_label, raw_conf) if raw_is_emerg else None)
            votes   = [x for x in detection_window if x is not None]
            n_votes = len(votes)

            if n_votes >= VOTE_MIN:
                best_vote  = max(votes, key=lambda x: x[1])
                label      = best_vote[0]
                confidence = best_vote[1]
                is_emergency = True
            else:
                label        = raw_label
                confidence   = raw_conf
                is_emergency = False

            display_colour, time_remaining = get_current_cycle_colour(cycle_start)

            # ── State machine (identical logic to camera.py) ───────────────
            if det_mode == "NORMAL":
                if esp:
                    send_signal(esp, display_colour)

                if display_colour == "RED" and is_emergency and not prev_is_emerg:
                    det_event_times = [now]
                    det_mode = "CONFIRMING"

            elif det_mode == "CONFIRMING":
                if display_colour != "RED":
                    det_mode = "NORMAL"
                    det_event_times = []
                    if esp:
                        send_signal(esp, display_colour, force=True)

                elif is_emergency and not prev_is_emerg:
                    gap = now - det_event_times[-1]
                    if gap <= DETECTION_GAP_MAX:
                        det_event_times.append(now)
                        if len(det_event_times) >= MIN_DETECTIONS:
                            det_mode        = "EMERGENCY"
                            emergency_start = now
                            if esp:
                                send_signal(esp, "EMERGENCY")
                            display_colour  = "GREEN"
                            det_event_times = []
                    else:
                        det_event_times = [now]

                elif not is_emergency and det_event_times:
                    if (now - det_event_times[-1]) > DETECTION_GAP_MAX:
                        det_mode = "NORMAL"
                        det_event_times = []
                        if esp:
                            send_signal(esp, display_colour, force=True)

            elif det_mode == "EMERGENCY":
                display_colour = "GREEN"
                time_remaining = max(0.0, EMERGENCY_GREEN_SECS - (now - emergency_start))

                if (now - emergency_start) >= EMERGENCY_GREEN_SECS:
                    det_mode        = "NORMAL"
                    emergency_start = None
                    det_event_times = []
                    detection_window.clear()
                    current_colour, _ = get_current_cycle_colour(cycle_start)
                    if esp:
                        send_signal(esp, current_colour, force=True)
                else:
                    if esp:
                        send_signal(esp, "EMERGENCY")

            # Progress bar
            if det_mode == "CONFIRMING" and det_event_times:
                hold_progress = len(det_event_times) / MIN_DETECTIONS
            elif det_mode == "EMERGENCY" and emergency_start:
                hold_progress = min(1.0, (now - emergency_start) / EMERGENCY_GREEN_SECS)
            else:
                hold_progress = 0.0

            frame = draw_overlay(frame, label, confidence, is_emergency,
                                 det_mode, hold_progress, display_colour, time_remaining)

            with lock:
                state.update({
                    "label":          label,
                    "confidence":     confidence,
                    "is_emergency":   is_emergency,
                    "mode":           det_mode,
                    "display_colour": display_colour,
                    "time_remaining": time_remaining,
                    "hold_progress":  hold_progress,
                })

            prev_is_emerg = is_emergency

        # ── Encode frame as JPEG ───────────────────────────────────────────
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with frame_lock:
                latest_jpeg = buf.tobytes()


# ── MJPEG generator ────────────────────────────────────────────────────────
def mjpeg_generator():
    while True:
        with frame_lock:
            jpg = latest_jpeg
        if jpg is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
            )
        time.sleep(0.033)   # ~30 FPS cap


# ── Startup ────────────────────────────────────────────────────────────────
@app.on_event("startup")
def startup():
    init_esp()
    t = threading.Thread(target=camera_thread, daemon=True)
    t.start()
    print("[server] Camera thread started.")


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/status")
def get_status():
    with lock:
        return JSONResponse(content={
            "app_mode":       state["app_mode"],
            "mode":           state["mode"],
            "signal":         state["display_colour"],
            "label":          state["label"],
            "confidence":     round(state["confidence"] * 100, 1),
            "is_emergency":   state["is_emergency"],
            "time_remaining": round(state["time_remaining"], 1),
            "hold_progress":  round(state["hold_progress"], 3),
        })


class ModeRequest(BaseModel):
    mode: str   # "AUTO" | "MANUAL"

@app.post("/set_mode")
def set_mode(req: ModeRequest):
    m = req.mode.upper()
    if m not in ("AUTO", "MANUAL"):
        return JSONResponse(status_code=400, content={"error": "mode must be AUTO or MANUAL"})
    with lock:
        state["app_mode"] = m
    return {"ok": True, "app_mode": m}


class SignalRequest(BaseModel):
    signal: str  # "RED" | "YELLOW" | "GREEN"

@app.post("/set_signal")
def set_signal(req: SignalRequest):
    s = req.signal.upper()
    if s not in ("RED", "YELLOW", "GREEN"):
        return JSONResponse(status_code=400, content={"error": "signal must be RED, YELLOW or GREEN"})
    with lock:
        state["manual_signal"]  = s
        state["display_colour"] = s
    # ESP32 will be updated on next camera loop iteration
    return {"ok": True, "signal": s}
