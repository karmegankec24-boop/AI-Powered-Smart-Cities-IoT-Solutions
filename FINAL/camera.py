import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import serial
import serial.tools.list_ports
import time

# ─── CONFIG ───────────────────────────────────────────────────────────────
import os as _os
# Use fine-tuned model if training has been done, otherwise fall back to COCO
_FINETUNED  = "best_model.pt"
_COCO_BASE  = "yolov8n.pt"
MODEL_PATH  = _FINETUNED if _os.path.exists(_FINETUNED) else _COCO_BASE

CONFIDENCE_THRESHOLD = 0.32             # raised: blocks weak ghost detections from motion

# 3-sighting confirmation rule:
#   Sight 1 detected  →  gap ≤ 3s  →  Sight 2 detected  →  gap ≤ 3s  →  Sight 3  →  EMERGENCY GREEN
DETECTION_GAP_MAX    = 3.0    # max allowed seconds between consecutive sightings
MIN_DETECTIONS       = 3     # 3 sightings required to confirm emergency
EMERGENCY_GREEN_SECS = 10.0  # hold GREEN for exactly 10s after confirmation, then resume normal

# Rolling-window vote (per-frame smoothing so motion blur doesn’t create a false sighting)
VOTE_WINDOW    = 10    # look at last 10 frames
VOTE_MIN       = 5     # 5/10 frames must agree - blocks brief motion false-triggers

# Emergency vehicle class names (lowercase comparison – handles any capitalisation)
EMERGENCY_CLASSES = {"ambulance", "fire truck", "emergency-vehicle", "emergency vehicle"}
# ───────────────────────────────────────────────────────────────────────────

NORMAL_CYCLE = [
    ("RED",    20),
    ("YELLOW",  5),
    ("GREEN",  10),
]


# ─── AUTO-DETECT ESP32 PORT ───────────────────────────────────────────────
def find_esp32_port():
    ports = serial.tools.list_ports.comports()
    print("\n--- Available Serial Ports ---")
    for p in ports:
        print(f"  {p.device}  ->  {p.description}")
    print("------------------------------\n")
    for p in ports:
        desc = p.description.lower()
        if any(k in desc for k in ["cp210", "ch340", "ch341", "uart", "esp", "usb serial"]):
            print(f"Auto-detected ESP32 on: {p.device}")
            return p.device
    print("Could not auto-detect. Falling back to COM5.")
    return "COM5"


# ─── YOLO INFERENCE HELPER ────────────────────────────────────────────────
def run_yolo(model, frame):
    """Detect emergency vehicles, draw boxes on frame, return (label, conf, is_emergency)."""
    results = model(frame, verbose=False)[0]
    best_label, best_conf = "No vehicle", 0.0

    for box in results.boxes:
        conf  = float(box.conf[0])
        cls   = int(box.cls[0])
        cname = model.names[cls].lower().strip()

        if cname not in EMERGENCY_CLASSES:
            continue  # skip non-emergency classes entirely

        # Track the best (highest confidence) detection
        if conf > best_conf:
            best_conf  = conf
            best_label = model.names[cls].title()   # e.g. "Ambulance", "Fire Truck"

        # ── Draw bounding box ──────────────────────────────────────────────
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if conf >= CONFIDENCE_THRESHOLD:
            # Confirmed detection → bright green box
            box_color  = (0, 220, 80)   # green
            txt_color  = (0, 0, 0)
            thickness  = 3
        else:
            # Seen but below threshold → dim yellow (still draw so user can see)
            box_color  = (0, 200, 220)  # yellow
            txt_color  = (0, 0, 0)
            thickness  = 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)

        # Label background + text
        label_text = f"{model.names[cls].title()}  {conf * 100:.1f}%"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_DUPLEX, 0.75, 2)
        tag_y = max(y1 - 4, th + 4)
        cv2.rectangle(frame, (x1, tag_y - th - 6), (x1 + tw + 6, tag_y + 2), box_color, -1)
        cv2.putText(frame, label_text, (x1 + 3, tag_y - 2),
                    cv2.FONT_HERSHEY_DUPLEX, 0.75, txt_color, 2, cv2.LINE_AA)

    is_emerg = best_conf >= CONFIDENCE_THRESHOLD
    return best_label, best_conf, is_emerg


# ─── SEND SIGNAL TO ESP32 ─────────────────────────────────────────────────
_last_sent = None

def send_signal(esp, command, force=False):
    global _last_sent
    if command == _last_sent and not force:
        return
    try:
        esp.write((command + "\n").encode())
        esp.flush()
        _last_sent = command
        print(f"[{time.strftime('%H:%M:%S')}] >>> Sent to ESP32: {command}")
    except Exception as e:
        print(f"[ERROR] Serial write failed: {e}")


# ─── GET CURRENT CYCLE COLOUR + TIME REMAINING ────────────────────────────
def get_current_cycle_colour(cycle_start_time):
    total_cycle = sum(d for _, d in NORMAL_CYCLE)
    elapsed = (time.time() - cycle_start_time) % total_cycle
    t = 0
    for colour, duration in NORMAL_CYCLE:
        t += duration
        if elapsed < t:
            return colour, t - elapsed
    return NORMAL_CYCLE[0][0], NORMAL_CYCLE[0][1]


# ─── DRAW OVERLAY ON FRAME ────────────────────────────────────────────────
def draw_overlay(frame, label, confidence, is_emergency,
                 mode, hold_progress, display_colour, time_remaining):
    h, w = frame.shape[:2]

    # Dark banner at top
    banner = frame.copy()
    cv2.rectangle(banner, (0, 0), (w, 110), (0, 0, 0), -1)
    cv2.addWeighted(banner, 0.6, frame, 0.4, 0, frame)

    # Line 1 - detection label
    if display_colour == "RED" or mode in ("CONFIRMING", "WAITING_CYCLE", "EMERGENCY"):
        if is_emergency:
            det_color = (0, 220, 80) if "Ambulance" in label else (30, 130, 255)
            det_text  = f"{label}  {confidence * 100:.1f}%"
        else:
            det_color = (160, 160, 160)
            det_text  = "Scanning... No emergency vehicle"
    else:
        det_color = (80, 80, 80)
        det_text  = f"Signal {display_colour} - detection paused"
    cv2.putText(frame, det_text, (18, 35),
                cv2.FONT_HERSHEY_DUPLEX, 0.90, det_color, 2, cv2.LINE_AA)

    # Line 2 - signal time remaining
    SIGNAL_TEXT_COLORS = {
        "RED":    (80,  80, 220),
        "GREEN":  (0,  200,  60),
        "YELLOW": (0,  200, 220),
    }
    cv2.putText(frame, f"{display_colour} remaining: {time_remaining:.1f}s",
                (18, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                SIGNAL_TEXT_COLORS.get(display_colour, (160, 160, 160)), 1, cv2.LINE_AA)

    # Line 3 - mode + signal
    MODE_COLORS = {
        "NORMAL":        (160, 160, 160),
        "CONFIRMING":    (0,   200, 230),
        "WAITING_CYCLE": (0,   200, 230),
        "EMERGENCY":     (0,   220,  80),
    }
    SIGNAL_COLORS = {
        "RED":    (60,  60, 220),
        "GREEN":  (0,  200,  60),
        "YELLOW": (0,  200, 220),
    }
    mode_col   = MODE_COLORS.get(mode, (160, 160, 160))
    signal_col = SIGNAL_COLORS.get(display_colour, (160, 160, 160))
    status_col = signal_col if mode == "NORMAL" else mode_col

    cv2.putText(frame, f"Mode: {mode}   Signal: {display_colour}",
                (18, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.60, status_col, 2, cv2.LINE_AA)

    # Coloured border matching signal
    cv2.rectangle(frame, (0, 0), (w-1, h-1), signal_col, 5)

    # Bottom status bar
    if mode == "CONFIRMING":
        bar_w = int((w - 40) * min(hold_progress, 1.0))
        cv2.rectangle(frame, (20, h-26), (w-20, h-10), (30, 30, 30), -1)
        cv2.rectangle(frame, (20, h-26), (20+bar_w, h-10), (0, 200, 230), -1)
        pct = int(min(hold_progress, 1.0) * 100)
        cv2.putText(frame, f"Sighting {int(hold_progress * MIN_DETECTIONS)}/{MIN_DETECTIONS} detected  (gap ≤ {DETECTION_GAP_MAX:.0f}s each)",
                    (22, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 230), 1, cv2.LINE_AA)
    elif mode == "WAITING_CYCLE":
        cv2.rectangle(frame, (20, h-26), (w-20, h-10), (30, 30, 60), -1)
        cv2.putText(frame, "Waiting for another sighting on RED...",
                    (22, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 220), 1, cv2.LINE_AA)
    elif mode == "EMERGENCY":
        cv2.rectangle(frame, (20, h-26), (w-20, h-10), (0, 180, 60), -1)
        cv2.putText(frame, "EMERGENCY GREEN - Ambulance / Fire engine in frame",
                    (22, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 100), 1, cv2.LINE_AA)
    else:
        cv2.rectangle(frame, (20, h-26), (w-20, h-10), (30, 30, 30), -1)
        if display_colour in ("GREEN", "YELLOW"):
            cv2.putText(frame, "Normal cycle - detection only active on RED",
                        (22, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Normal cycle running - watching for emergency vehicle",
                        (22, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

    return frame


# ─── MAIN ─────────────────────────────────────────────────────────────────
def main():
    # Load YOLOv8 model
    print(f"Loading YOLOv8 model: {MODEL_PATH} ...")
    if MODEL_PATH == _FINETUNED:
        print("  Using fine-tuned model (best_model.pt) ✓")
    else:
        print("  best_model.pt not found – using base yolov8n.pt.")
        print("  Run  python quick_train.py  for better accuracy.")
    model = YOLO(MODEL_PATH)
    print(f"YOLOv8 loaded. Emergency classes: {EMERGENCY_CLASSES}")

    # Connect to ESP32
    port = find_esp32_port()
    try:
        esp = serial.Serial(port, 115200, timeout=1)
        time.sleep(2)
        print(f"Connected to ESP32 on {port} at 115200 baud.\n")
    except serial.SerialException as e:
        print(f"ERROR opening {port}: {e}")
        print("Make sure Arduino Serial Monitor is closed, then retry.")
        return

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("Camera opened. Press Q to quit.\n")

    # ── State variables ────────────────────────────────────────────────────
    mode            = "NORMAL"
    det_event_times = []         # timestamps of each distinct sighting event (on RED)
    prev_is_emerg   = False      # previous frame's emergency status (for edge detection)
    emergency_start = None       # time when EMERGENCY GREEN was locked (for 10s timer)
    cycle_start     = time.time()  # set once, NEVER reset – timer always keeps running

    # ── Rolling detection window (10 frames) ───────────────────────────
    # Each entry: (label, conf) if detected above threshold on that frame, else None
    detection_window = deque(maxlen=VOTE_WINDOW)

    # Tell ESP32 what colour to show (Python is the timer source of truth)
    send_signal(esp, "RED", force=True)
    # ───────────────────────────────────────────────────────────────────────

    while True:
        ret, frame = cap.read()
        if not ret:
            print("ERROR: Failed to grab frame.")
            break

        now = time.time()

        # ── Run YOLOv8 inference on current frame ─────────────────────
        raw_label, raw_conf, raw_is_emerg = run_yolo(model, frame)

        # ── Rolling window vote: push this frame's result, then tally ─────────
        detection_window.append((raw_label, raw_conf) if raw_is_emerg else None)

        # Count how many of the last VOTE_WINDOW frames voted 'detected'
        votes   = [x for x in detection_window if x is not None]
        n_votes = len(votes)

        if n_votes >= VOTE_MIN:
            # Enough frames agree – pick the highest-confidence label from the window
            best_vote  = max(votes, key=lambda x: x[1])
            label      = best_vote[0]
            confidence = best_vote[1]
            is_emergency = True
        else:
            label        = raw_label
            confidence   = raw_conf
            is_emergency = False

        # Get current signal colour and how many seconds remain
        display_colour, time_remaining = get_current_cycle_colour(cycle_start)

        # ── State machine ──────────────────────────────────────────────────

        if mode == "NORMAL":
            # Keep ESP32 LED in sync with the traffic cycle
            send_signal(esp, display_colour)

            # Only act on RED signal
            if display_colour == "RED" and is_emergency and not prev_is_emerg:
                # New sighting edge (vehicle just came into view)
                det_event_times = [now]
                mode = "CONFIRMING"
                print(f"[{time.strftime('%H:%M:%S')}] Sighting 1/{MIN_DETECTIONS} on RED "
                      f"({time_remaining:.1f}s left) – need {MIN_DETECTIONS-1} more within {DETECTION_GAP_MAX}s each")

        elif mode == "CONFIRMING":
            # Waiting for MIN_DETECTIONS sightings, each gap ≤ DETECTION_GAP_MAX seconds

            # Check if the signal changed off RED – if so, reset and go back to NORMAL
            if display_colour != "RED":
                mode = "NORMAL"
                det_event_times = []
                send_signal(esp, display_colour, force=True)

            # New sighting edge (vehicle just re-appeared)
            elif is_emergency and not prev_is_emerg:
                gap = now - det_event_times[-1]
                if gap <= DETECTION_GAP_MAX:
                    det_event_times.append(now)
                    n = len(det_event_times)
                    print(f"[{time.strftime('%H:%M:%S')}] Sighting {n}/{MIN_DETECTIONS} "
                          f"(gap {gap:.1f}s ≤ {DETECTION_GAP_MAX}s)")
                    if n >= MIN_DETECTIONS:
                        # All 3 sightings confirmed → EMERGENCY GREEN for 10s
                        mode            = "EMERGENCY"
                        emergency_start = now
                        send_signal(esp, "EMERGENCY")
                        display_colour  = "GREEN"
                        det_event_times = []
                        print(f"[{time.strftime('%H:%M:%S')}] CONFIRMED – {MIN_DETECTIONS} sightings "
                              f"→ EMERGENCY GREEN for {EMERGENCY_GREEN_SECS:.0f}s!")
                else:
                    # Gap too large – this sighting restarts the chain
                    det_event_times = [now]
                    print(f"[{time.strftime('%H:%M:%S')}] Gap {gap:.1f}s > {DETECTION_GAP_MAX}s "
                          f"– restarting from sighting 1/{MIN_DETECTIONS}")

            # No detection for too long – lose interest and go back to NORMAL
            elif not is_emergency and det_event_times:
                if (now - det_event_times[-1]) > DETECTION_GAP_MAX:
                    mode = "NORMAL"
                    det_event_times = []
                    send_signal(esp, display_colour, force=True)
                    print(f"[{time.strftime('%H:%M:%S')}] No sighting for >{DETECTION_GAP_MAX}s "
                          f"– back to NORMAL")

        elif mode == "EMERGENCY":
            display_colour = "GREEN"
            time_remaining = max(0.0, EMERGENCY_GREEN_SECS - (now - emergency_start))

            if (now - emergency_start) >= EMERGENCY_GREEN_SECS:
                # 10 seconds elapsed → exit EMERGENCY, resume normal cycle
                mode            = "NORMAL"
                emergency_start = None
                det_event_times = []
                detection_window.clear()          # flush stale votes so next detection starts fresh
                current_colour, _ = get_current_cycle_colour(cycle_start)
                send_signal(esp, current_colour, force=True)   # ONE clean signal to ESP32
                print(f"[{time.strftime('%H:%M:%S')}] Emergency GREEN ended (10s) → resuming normal cycle")
            else:
                send_signal(esp, "EMERGENCY")     # keep GREEN only while still in EMERGENCY


        # ── Progress bar calculation ───────────────────────────────────────
        if mode == "CONFIRMING" and det_event_times:
            hold_progress = len(det_event_times) / MIN_DETECTIONS
        elif mode == "EMERGENCY" and emergency_start:
            hold_progress = min(1.0, (now - emergency_start) / EMERGENCY_GREEN_SECS)
        else:
            hold_progress = 0.0

        # ── Draw overlay and show frame ────────────────────────────────────
        frame = draw_overlay(frame, label, confidence,
                             is_emergency, mode, hold_progress,
                             display_colour, time_remaining)
        cv2.imshow("Emergency Vehicle Detection  |  Press Q to quit", frame)

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quitting...")
            break

        prev_is_emerg = is_emergency   # track edge for next frame

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    esp.close()
    print("Done.")


# ─── ENTRY POINT ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()