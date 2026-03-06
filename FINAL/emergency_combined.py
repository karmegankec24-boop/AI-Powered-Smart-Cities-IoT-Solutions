"""
emergency_combined.py
=====================
Place this file INSIDE the FINAL folder.

Runs in ONE terminal. Does everything simultaneously:
  Audio  : Listens for siren / traffic+siren  (TFLite model)
  Visual : Detects ambulance / fire engine     (YOLOv8)
  Logic  : BOTH must trigger during RED signal
           -> holds for 2 seconds -> switches to GREEN

SETTINGS:
  Camera : Forces EXTERNAL webcam  (skips index 0 = built-in laptop camera)
  Audio  : Forces LAPTOP MICROPHONE (OS default input device)
  Audio  : Siren confidence must stay ABOVE its threshold continuously
           for AUDIO_SUSTAIN_SECS = 3 seconds before audio is confirmed.
           If confidence drops below threshold even once, the timer
           resets fully to 0 and must build back up from scratch.
"""

import cv2
import threading
import time
import os
import serial
import serial.tools.list_ports
from collections import deque
from datetime import datetime
from ultralytics import YOLO

# ==============================================================================
#  PATHS
# ==============================================================================
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
SIREN_DIR    = r"C:\Users\jaswa\OneDrive\Desktop\combination of two files\siren_detector"
TFLITE_MODEL = os.path.join(SIREN_DIR, "soundclassifier_with_metadata.tflite")
VISUAL_MODEL = os.path.join(BASE_DIR,  "best_model.pt")
if not os.path.exists(VISUAL_MODEL):
    VISUAL_MODEL = "yolov8n.pt"
LOG_FILE     = os.path.join(SIREN_DIR, "detections.log")

# ==============================================================================
#  TRAFFIC LIGHT CONFIG
# ==============================================================================
NORMAL_CYCLE = [("RED", 20), ("YELLOW", 5), ("GREEN", 10)]

# ==============================================================================
#  VISUAL DETECTION CONFIG
# ==============================================================================
CONFIDENCE_THRESHOLD = 0.60
EMERGENCY_CLASSES    = {"ambulance", "fire truck", "fire engine", "emergency-vehicle"}
VOTE_WINDOW          = 10
VOTE_MIN             = 5

# ==============================================================================
#  AUDIO DETECTION CONFIG
# ==============================================================================
SAMPLE_RATE   = 16000
AUDIO_SAMPLES = 44032

SIREN_CONF = {
    1: 0.90,   # Siren only        -> must be >= 90%
    2: 0.65,   # Traffic + siren   -> must be >= 65%
}

ALL_AUDIO_CLASSES = {
    0: "Background Noise",
    1: "Siren only",
    2: "Traffic noise+siren",
}

# Siren must stay above threshold continuously for this many seconds
# If it drops below even once -> timer resets to 0
AUDIO_SUSTAIN_SECS = 3.0

# ==============================================================================
#  COMBINED TRIGGER CONFIG
# ==============================================================================
BOTH_DETECTED_SECS   = 2.0
EMERGENCY_GREEN_SECS = 10.0

# ==============================================================================
#  MIC SELECTION
#  None  -> OS default mic (laptop built-in mic)
#  0,1,2 -> force a specific device index
#  Run: python -c "import sounddevice as sd; print(sd.query_devices())"
# ==============================================================================
MIC_DEVICE_INDEX = None

# ==============================================================================
#  SHARED STATE
# ==============================================================================
audio_siren_active    = False
audio_sustain_elapsed = 0.0
audio_lock            = threading.Lock()
sustain_lock          = threading.Lock()

# Manual mode state
manual_mode    = False       # True = manual, False = auto
manual_signal  = "RED"       # holds the user-chosen signal in manual mode


# ==============================================================================
#  AUDIO THREAD
# ==============================================================================
def audio_thread_fn(stop_event):
    global audio_siren_active, audio_sustain_elapsed

    try:
        import tensorflow as tf
        import sounddevice as sd
    except ImportError as e:
        print(f"[AUDIO] Missing library: {e}")
        print("[AUDIO] Run:  pip install tensorflow sounddevice")
        return

    if not os.path.exists(TFLITE_MODEL):
        print(f"[AUDIO] TFLite model not found: {TFLITE_MODEL}")
        return

    print("\n[AUDIO] Available input devices:")
    try:
        devices    = sd.query_devices()
        default_in = sd.query_devices(kind="input")
        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0:
                is_selected = (
                    (MIC_DEVICE_INDEX is None and d["name"] == default_in["name"])
                    or i == MIC_DEVICE_INDEX
                )
                tag = "  << SELECTED" if is_selected else ""
                print(f"         [{i}] {d['name']}{tag}")
        print(f"[AUDIO] OS default input : {default_in['name']}")
        if MIC_DEVICE_INDEX is not None:
            forced = sd.query_devices(MIC_DEVICE_INDEX)
            print(f"[AUDIO] Forced device [{MIC_DEVICE_INDEX}] : {forced['name']}")
    except Exception as e:
        print(f"[AUDIO] Could not list audio devices: {e}")
    print()

    try:
        interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL)
        interpreter.allocate_tensors()
        inp_details = interpreter.get_input_details()
        out_details = interpreter.get_output_details()
        print(f"[AUDIO] TFLite model loaded successfully.")
        print(f"[AUDIO] Siren must stay above threshold for {AUDIO_SUSTAIN_SECS:.0f}s to confirm.")
        print(f"[AUDIO] Timer resets to 0 if confidence drops below threshold even once.")
        print(f"[AUDIO] Listening on laptop microphone...\n")
    except Exception as e:
        print(f"[AUDIO] Could not load TFLite model: {e}")
        return

    siren_start_time = None
    currently_active = False

    while not stop_event.is_set():
        try:
            audio_data = sd.rec(
                AUDIO_SAMPLES,
                samplerate=SAMPLE_RATE,
                channels=1,
                device=MIC_DEVICE_INDEX,
                dtype="float32",
            )
            sd.wait()

            audio_flat  = audio_data.flatten().reshape(inp_details[0]["shape"])
            interpreter.set_tensor(inp_details[0]["index"], audio_flat)
            interpreter.invoke()
            predictions = interpreter.get_tensor(out_details[0]["index"])[0]

            best_cls    = int(predictions.argmax())
            best_conf   = float(predictions[best_cls])
            class_label = ALL_AUDIO_CLASSES.get(best_cls, f"Class {best_cls}")

            if best_cls == 0:
                passes_threshold = False
            elif best_cls in SIREN_CONF:
                passes_threshold = best_conf >= SIREN_CONF[best_cls]
            else:
                passes_threshold = False

            now = time.time()

            if passes_threshold:
                if siren_start_time is None:
                    siren_start_time = now
                    print(
                        f"[AUDIO] Candidate: {class_label} ({best_conf*100:.1f}%) "
                        f"-- sustain timer started (0.0s / {AUDIO_SUSTAIN_SECS:.0f}s)"
                    )

                elapsed   = now - siren_start_time
                remaining = max(0.0, AUDIO_SUSTAIN_SECS - elapsed)

                with sustain_lock:
                    audio_sustain_elapsed = elapsed

                if elapsed >= AUDIO_SUSTAIN_SECS:
                    if not currently_active:
                        currently_active = True
                        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        print(
                            f"[AUDIO] *** SIREN CONFIRMED after {AUDIO_SUSTAIN_SECS:.0f}s "
                            f"--> {class_label} ({best_conf*100:.1f}%) ***"
                        )
                        with open(LOG_FILE, "a") as f:
                            f.write(
                                f"{ts} | CONFIRMED {class_label} "
                                f"({best_conf*100:.1f}%) after {AUDIO_SUSTAIN_SECS:.0f}s sustained\n"
                            )
                        try:
                            import winsound
                            winsound.Beep(800, 200)
                        except Exception:
                            pass
                    else:
                        print(
                            f"[AUDIO] ACTIVE: {class_label} ({best_conf*100:.1f}%) "
                            f"| sustained {elapsed:.1f}s"
                        )
                else:
                    print(
                        f"[AUDIO] Confirming: {class_label} ({best_conf*100:.1f}%) "
                        f"-- {elapsed:.1f}s / {AUDIO_SUSTAIN_SECS:.0f}s "
                        f"({remaining:.1f}s remaining)"
                    )

            else:
                if siren_start_time is not None:
                    elapsed_so_far = now - siren_start_time
                    print(
                        f"[AUDIO] Dropped below threshold: {class_label} "
                        f"({best_conf*100:.1f}%) after {elapsed_so_far:.1f}s "
                        f"-- sustain timer RESET to 0."
                    )

                siren_start_time = None
                currently_active = False
                with sustain_lock:
                    audio_sustain_elapsed = 0.0
                print(f"[AUDIO] {class_label} ({best_conf*100:.1f}%) -- no siren")

            with audio_lock:
                audio_siren_active = currently_active

        except Exception as e:
            if not stop_event.is_set():
                print(f"[AUDIO] Error: {e}")

    print("[AUDIO] Audio thread stopped.")


# ==============================================================================
#  ESP32 HELPERS
# ==============================================================================
def find_esp32():
    for port in serial.tools.list_ports.comports():
        desc = port.description.lower()
        if any(k in desc for k in ["cp210", "ch340", "ch341", "uart", "esp", "usb serial"]):
            print(f"[ESP32] Auto-detected ESP32 on: {port.device}")
            return port.device
    print("[ESP32] ESP32 not found automatically, defaulting to COM5.")
    return "COM5"


_last_sent_cmd = [None]

def send_signal(esp, cmd, force=False):
    if cmd == _last_sent_cmd[0] and not force:
        return
    try:
        esp.write((cmd + "\n").encode())
        esp.flush()
        _last_sent_cmd[0] = cmd
        print(f"[ESP32] Sent: {cmd}")
    except Exception as e:
        print(f"[ESP32] Send error: {e}")


# ==============================================================================
#  TRAFFIC CYCLE HELPER
# ==============================================================================
def get_current_signal(cycle_start):
    total_duration = sum(dur for _, dur in NORMAL_CYCLE)
    elapsed        = (time.time() - cycle_start) % total_duration
    cumulative     = 0
    for colour, dur in NORMAL_CYCLE:
        cumulative += dur
        if elapsed < cumulative:
            return colour, cumulative - elapsed
    return NORMAL_CYCLE[0][0], NORMAL_CYCLE[0][1]


# ==============================================================================
#  YOLO VISUAL INFERENCE
# ==============================================================================
def run_yolo(model, frame):
    results   = model(frame, verbose=False)[0]
    best_lbl  = "No vehicle"
    best_conf = 0.0

    for box in results.boxes:
        conf  = float(box.conf[0])
        cls   = int(box.cls[0])
        cname = model.names[cls].lower().strip()

        if cname not in EMERGENCY_CLASSES:
            continue

        if conf > best_conf:
            best_conf = conf
            best_lbl  = model.names[cls].title()

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        colour = (0, 220, 80) if conf >= CONFIDENCE_THRESHOLD else (0, 200, 220)
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 3)

        label_text = f"{model.names[cls].title()}  {conf*100:.1f}%"
        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_DUPLEX, 0.75, 2)
        ty = max(y1 - 4, text_h + 4)
        cv2.rectangle(frame, (x1, ty - text_h - 6), (x1 + text_w + 6, ty + 2), colour, -1)
        cv2.putText(frame, label_text, (x1 + 3, ty - 2),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)

    return best_lbl, best_conf, best_conf >= CONFIDENCE_THRESHOLD


# ==============================================================================
#  HUD OVERLAY
# ==============================================================================
def draw_overlay(frame,
                 vis_label, vis_conf, vis_active,
                 aud_active, sustain_elapsed,
                 signal, time_rem,
                 mode, both_timer):
    h, w = frame.shape[:2]

    SIGNAL_COLOURS = {
        "RED":    (60,  60, 220),
        "GREEN":  (0,  200,  60),
        "YELLOW": (0,  200, 220),
    }
    signal_col = SIGNAL_COLOURS.get(signal, (160, 160, 160))

    banner = frame.copy()
    cv2.rectangle(banner, (0, 0), (w, 150), (0, 0, 0), -1)
    cv2.addWeighted(banner, 0.65, frame, 0.35, 0, frame)

    # Row 1 - Visual
    vcol = (0, 220, 80) if vis_active else (160, 160, 160)
    vtxt = (f"Camera: {vis_label}  {vis_conf*100:.1f}%"
            if vis_active else "Camera: Scanning for emergency vehicle...")
    cv2.putText(frame, vtxt, (14, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, vcol, 2, cv2.LINE_AA)

    # Row 2 - Audio
    if aud_active:
        acol = (0, 220, 80)
        atxt = f"Audio (Laptop Mic): SIREN CONFIRMED  ({AUDIO_SUSTAIN_SECS:.0f}s sustained)"
    elif 0 < sustain_elapsed < AUDIO_SUSTAIN_SECS:
        acol      = (0, 220, 220)
        remaining = AUDIO_SUSTAIN_SECS - sustain_elapsed
        atxt = (f"Audio (Laptop Mic): Confirming siren... "
                f"{sustain_elapsed:.1f}s / {AUDIO_SUSTAIN_SECS:.0f}s  "
                f"({remaining:.1f}s remaining)")
    else:
        acol = (160, 160, 160)
        atxt = "Audio (Laptop Mic): No siren detected"
    cv2.putText(frame, atxt, (14, 64),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, acol, 2, cv2.LINE_AA)

    # Row 3 - Sustain progress bar
    if 0 < sustain_elapsed < AUDIO_SUSTAIN_SECS:
        progress = sustain_elapsed / AUDIO_SUSTAIN_SECS
        bar_fill = int((w - 28) * progress)
        cv2.rectangle(frame, (14, 72), (w - 14, 86), (40, 40, 40), -1)
        cv2.rectangle(frame, (14, 72), (14 + bar_fill, 86), (0, 220, 220), -1)
        cv2.putText(frame, "sustain", (14, 98),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 80), 1, cv2.LINE_AA)

    # Row 4 - Signal info
    cv2.putText(frame,
                f"Signal: {signal}   {time_rem:.1f}s remaining   Mode: {mode}",
                (14, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.60, signal_col, 2, cv2.LINE_AA)

    # Bottom bar
    if mode == "WAITING_BOTH" and both_timer > 0:
        prog  = min(both_timer / BOTH_DETECTED_SECS, 1.0)
        bar_w = int((w - 40) * prog)
        cv2.rectangle(frame, (20, h - 30), (w - 20, h - 10), (30, 30, 30), -1)
        cv2.rectangle(frame, (20, h - 30), (20 + bar_w, h - 10), (0, 200, 230), -1)
        cv2.putText(frame,
                    f"Both detected! Confirming emergency...  "
                    f"{both_timer:.1f}s / {BOTH_DETECTED_SECS:.0f}s",
                    (22, h - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 200, 230), 1, cv2.LINE_AA)
    elif mode == "EMERGENCY":
        cv2.rectangle(frame, (20, h - 30), (w - 20, h - 10), (0, 140, 40), -1)
        cv2.putText(frame,
                    "EMERGENCY GREEN ACTIVE -- Ambulance / Fire Engine passage in progress",
                    (22, h - 34), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 100), 1, cv2.LINE_AA)
    else:
        cv2.rectangle(frame, (20, h - 30), (w - 20, h - 10), (30, 30, 30), -1)
        hint = ("Waiting: both camera AND audio must confirm simultaneously on RED"
                if mode == "NORMAL" else "Signal is not RED -- standing by for red phase")
        cv2.putText(frame, hint, (22, h - 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (100, 100, 100), 1, cv2.LINE_AA)

    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), signal_col, 6)
    return frame


# ==============================================================================
#  FIND EXTERNAL WEBCAM
# ==============================================================================
def find_external_webcam():
    ALLOW_BUILTIN_FALLBACK = False
    search_indices = [1, 2, 3]
    if ALLOW_BUILTIN_FALLBACK:
        search_indices.append(0)

    for idx in search_indices:
        label = "Built-in (index 0)" if idx == 0 else f"External webcam (index {idx})"
        print(f"[VISUAL] Trying {label}...")
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            print(f"[VISUAL]   -> Not available.")
            continue
        time.sleep(0.5)
        ret, test_frame = cap.read()
        if ret and test_frame is not None and test_frame.size > 0:
            print(f"[VISUAL]   -> Working! Using {label}.")
            return cap, idx
        cap.release()
        print(f"[VISUAL]   -> Opened but could not read a frame.")

    return None, -1


# ==============================================================================
#  MAIN
# ==============================================================================
def main():
    global manual_mode, manual_signal
    print("=" * 65)
    print("  Emergency Vehicle Combined Detector")
    print("=" * 65)
    print(f"  Camera  : EXTERNAL WEBCAM (indices 1-3, skips built-in at 0)")
    print(f"  Audio   : LAPTOP MICROPHONE (OS default input device)")
    print(f"  Sustain : Siren must stay >= threshold for {AUDIO_SUSTAIN_SECS:.0f}s to confirm")
    print(f"            (timer resets to 0 if confidence drops even once)")
    print(f"  Trigger : Camera + Audio BOTH confirmed on RED -> {BOTH_DETECTED_SECS:.0f}s hold -> GREEN")
    print(f"  Green   : Emergency GREEN held for {EMERGENCY_GREEN_SECS:.0f}s")
    print("=" * 65 + "\n")

    print(f"[VISUAL] Loading YOLO model: {VISUAL_MODEL}")
    model = YOLO(VISUAL_MODEL)
    print("[VISUAL] YOLO model ready.\n")

    port = find_esp32()
    try:
        esp = serial.Serial(port, 115200, timeout=1)
        time.sleep(2)
        print(f"[ESP32]  Connected on {port}\n")
    except serial.SerialException as e:
        print(f"[ESP32]  Connection failed: {e}")
        print(f"[ESP32]  Close Arduino Serial Monitor if open, then retry.")
        return

    print("[VISUAL] Searching for external webcam (skipping built-in at index 0)...")
    cap, camera_index = find_external_webcam()

    if cap is None:
        print("\n[VISUAL] No external webcam found on indices 1, 2, or 3.")
        print("[VISUAL] Make sure your USB webcam is plugged in and try again.")
        esp.close()
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("[VISUAL] Flushing startup frames...")
    for _ in range(15):
        cap.read()
    print(f"[VISUAL] External webcam ready (index {camera_index}). Press Q to quit.\n")

    stop_event   = threading.Event()
    audio_thread = threading.Thread(target=audio_thread_fn, args=(stop_event,), daemon=True)
    audio_thread.start()

    mode             = "NORMAL"
    cycle_start      = time.time()
    both_start       = None
    emergency_start  = None
    detection_window = deque(maxlen=VOTE_WINDOW)

    send_signal(esp, "RED", force=True)

    print("[MAIN] Running. Press Q in the camera window to stop.")
    print("[MAIN] Keys:  M = toggle Manual/Auto | R = Red | Y = Yellow | G = Green\n")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[VISUAL] Frame read failed -- check webcam connection.")
            break

        now = time.time()

        # Visual inference + voting
        raw_lbl, raw_conf, raw_active = run_yolo(model, frame)
        detection_window.append((raw_lbl, raw_conf) if raw_active else None)
        votes = [v for v in detection_window if v is not None]
        if len(votes) >= VOTE_MIN:
            best_vote           = max(votes, key=lambda v: v[1])
            vis_label, vis_conf = best_vote
            vis_active          = True
        else:
            vis_label, vis_conf = raw_lbl, raw_conf
            vis_active          = False

        # Read shared audio state
        with audio_lock:
            aud_active = audio_siren_active
        with sustain_lock:
            sustain_elapsed = audio_sustain_elapsed

        # Traffic signal (auto vs manual)
        if manual_mode:
            signal   = manual_signal
            time_rem = 0.0
        else:
            signal, time_rem = get_current_signal(cycle_start)

        # State machine (auto only)
        if manual_mode:
            # In manual mode: send chosen signal, skip emergency logic
            send_signal(esp, signal)
            mode       = "NORMAL"
            both_start = None
        else:
            if mode == "NORMAL":
                send_signal(esp, signal)
                if signal == "RED" and vis_active and aud_active:
                    both_start = now
                    mode       = "WAITING_BOTH"
                    print(f"[LOGIC] Both confirmed on RED! Holding {BOTH_DETECTED_SECS:.0f}s before GREEN...")

            elif mode == "WAITING_BOTH":
                elapsed_both = now - both_start
                if signal != "RED":
                    mode = "NORMAL"; both_start = None
                    send_signal(esp, signal, force=True)
                    print("[LOGIC] Signal left RED -- reset.")
                elif not (vis_active and aud_active):
                    mode = "NORMAL"; both_start = None
                    send_signal(esp, signal, force=True)
                    print("[LOGIC] One detector lost -- reset.")
                elif elapsed_both >= BOTH_DETECTED_SECS:
                    mode            = "EMERGENCY"
                    emergency_start = now
                    both_start      = None
                    send_signal(esp, "EMERGENCY", force=True)
                    detection_window.clear()
                    print(f"[LOGIC] EMERGENCY CONFIRMED! GREEN for {EMERGENCY_GREEN_SECS:.0f}s.")

            elif mode == "EMERGENCY":
                signal   = "GREEN"
                time_rem = max(0.0, EMERGENCY_GREEN_SECS - (now - emergency_start))
                send_signal(esp, "EMERGENCY")
                if (now - emergency_start) >= EMERGENCY_GREEN_SECS:
                    mode            = "NORMAL"
                    emergency_start = None
                    detection_window.clear()
                    cur, _ = get_current_signal(cycle_start)
                    send_signal(esp, cur, force=True)
                    print("[LOGIC] Emergency GREEN ended -- resuming normal cycle.")

        # Draw and display
        both_elapsed = (now - both_start) if both_start else 0.0

        # Manual mode overlay: draw a prominent banner
        if manual_mode:
            SIGNAL_COLOURS_CV = {
                "RED":    (60,  60, 220),
                "GREEN":  (0,  200,  60),
                "YELLOW": (0,  200, 220),
            }
            sc = SIGNAL_COLOURS_CV.get(signal, (160, 160, 160))
            h_fr, w_fr = frame.shape[:2]
            banner = frame.copy()
            cv2.rectangle(banner, (0, 0), (w_fr, 130), (0, 0, 0), -1)
            cv2.addWeighted(banner, 0.7, frame, 0.3, 0, frame)
            cv2.putText(frame, f"MANUAL MODE  |  Signal: {signal}",
                        (14, 38), cv2.FONT_HERSHEY_DUPLEX, 0.90, sc, 2, cv2.LINE_AA)
            cv2.putText(frame, "Keys: M=toggle Auto/Manual   R=Red   Y=Yellow   G=Green",
                        (14, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (160, 160, 160), 1, cv2.LINE_AA)

            # Audio row (still shown in manual mode)
            with audio_lock:
                aud_a = audio_siren_active
            with sustain_lock:
                sust_e = audio_sustain_elapsed
            if aud_a:
                atxt = "Audio: SIREN CONFIRMED"
                acol = (0, 220, 80)
            elif 0 < sust_e < AUDIO_SUSTAIN_SECS:
                atxt = f"Audio: Confirming siren... {sust_e:.1f}s / {AUDIO_SUSTAIN_SECS:.0f}s"
                acol = (0, 220, 220)
            else:
                atxt = "Audio: No siren detected"
                acol = (160, 160, 160)
            cv2.putText(frame, atxt, (14, 98),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, acol, 2, cv2.LINE_AA)
            cv2.rectangle(frame, (0, 0), (w_fr - 1, h_fr - 1), sc, 6)
        else:
            frame = draw_overlay(frame,
                                 vis_label, vis_conf, vis_active,
                                 aud_active, sustain_elapsed,
                                 signal, time_rem,
                                 mode, both_elapsed)

        cv2.imshow("Emergency Detector -- External Webcam + Laptop Mic | Q to quit", frame)

        # Handle keypresses
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("\n[MAIN] Q pressed -- shutting down...")
            break
        elif key == ord("m") or key == ord("M"):
            manual_mode = not manual_mode
            label = "MANUAL" if manual_mode else "AUTO"
            print(f"[MANUAL] Switched to {label} mode.")
            if manual_mode:
                print(f"[MANUAL] Signal locked to {manual_signal}. Press R/Y/G to change.")
            # Reset auto state when switching back
            if not manual_mode:
                mode            = "NORMAL"
                both_start      = None
                emergency_start = None
                detection_window.clear()
                cycle_start = time.time()
        elif manual_mode:
            if key == ord("r") or key == ord("R"):
                manual_signal = "RED"
                print(f"[MANUAL] Signal -> RED")
            elif key == ord("y") or key == ord("Y"):
                manual_signal = "YELLOW"
                print(f"[MANUAL] Signal -> YELLOW")
            elif key == ord("g") or key == ord("G"):
                manual_signal = "GREEN"
                print(f"[MANUAL] Signal -> GREEN")

    stop_event.set()
    cap.release()
    cv2.destroyAllWindows()
    esp.close()
    audio_thread.join(timeout=5)
    print("All components stopped. Goodbye!")


if __name__ == "__main__":
    main()