import collections
import threading
import numpy as np
import sounddevice as sd

try:
    import tflite_runtime.interpreter as tflite
    Interpreter = tflite.Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False

# ── SETTINGS ──────────────────────────────────────────────
MODEL_PATH    = "soundclassifier_with_metadata.tflite"
SAMPLE_RATE   = 16000
WINDOW_SEC    = 0.975
CONFIDENCE_TH = 0.50
HOLD_SEC      = 2.0
PIN_RED       = 17
PIN_YELLOW    = 27
PIN_GREEN     = 22

# YOUR LABELS
CLASS_AMBULANCE_TRAFFIC = 0   # Ambulance siren sound with traffic noise
CLASS_BACKGROUND        = 1   # Background Noise
CLASS_SIREN             = 2   # Only siren

LABEL_NAMES = {
    CLASS_AMBULANCE_TRAFFIC: "Ambulance + Traffic",
    CLASS_BACKGROUND:        "Background Noise",
    CLASS_SIREN:             "Only Siren",
}
# ──────────────────────────────────────────────────────────

def setup_gpio():
    if not GPIO_AVAILABLE:
        return
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    for pin in [PIN_RED, PIN_YELLOW, PIN_GREEN]:
        GPIO.setup(pin, GPIO.OUT)
        GPIO.output(pin, GPIO.LOW)

def set_led(colour):
    states = {
        "red":    (True,  False, False),
        "yellow": (False, True,  False),
        "green":  (False, False, True),
        "off":    (False, False, False),
    }
    r, y, g = states.get(colour, (False, False, False))
    if GPIO_AVAILABLE:
        GPIO.output(PIN_RED,    GPIO.HIGH if r else GPIO.LOW)
        GPIO.output(PIN_YELLOW, GPIO.HIGH if y else GPIO.LOW)
        GPIO.output(PIN_GREEN,  GPIO.HIGH if g else GPIO.LOW)
    icons = {"red": "🔴 RED", "yellow": "🟡 YELLOW", "green": "🟢 GREEN"}
    print(f"  LED -> {icons.get(colour, colour)}")

def cleanup_gpio():
    if GPIO_AVAILABLE:
        set_led("off")
        GPIO.cleanup()

def load_model():
    interp = Interpreter(model_path=MODEL_PATH)
    interp.allocate_tensors()
    return interp, interp.get_input_details()[0], interp.get_output_details()[0]

def run_inference(interp, inp, out, audio):
    expected = int(np.prod(inp['shape']))
    audio = audio.flatten().astype(np.float32)
    if len(audio) < expected:
        audio = np.pad(audio, (0, expected - len(audio)))
    else:
        audio = audio[:expected]
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    if inp['dtype'] == np.int16:
        audio = (audio * 32767).astype(np.int16)
    audio = audio.reshape(inp['shape'])
    interp.set_tensor(inp['index'], audio)
    interp.invoke()
    scores = interp.get_tensor(out['index'])[0]
    class_id = int(np.argmax(scores))
    return class_id, float(scores[class_id])

class SignalController:
    def __init__(self):
        self.window_size = max(1, int(HOLD_SEC / WINDOW_SEC))
        self.history     = collections.deque(maxlen=self.window_size)
        self.current_led = "yellow"
        set_led("yellow")
        print(f"Traffic light started on YELLOW")
        print(f"Needs {self.window_size} detections in a row (~2 seconds) to switch\n")

    def update(self, class_id, conf):
        self.history.append(class_id)
        if len(self.history) < self.window_size:
            return
        if class_id is None:
            return
        if all(c == class_id for c in self.history):
            self._apply(class_id, conf)

    def _apply(self, class_id, conf):
        pct = f"{conf*100:.1f}%"
        if class_id == CLASS_BACKGROUND:
            if self.current_led != "red":
                print(f"\n[{pct}] Background Noise detected for 2s -> Changing signal to RED")
                self.current_led = "red"
                set_led("red")
            else:
                print(f"[{pct}] Background Noise – signal stays RED")
        elif class_id == CLASS_SIREN:
            if self.current_led != "green":
                print(f"\n[{pct}] Only Siren detected for 2s -> Current signal (class 2) changed to GREEN")
                self.current_led = "green"
                set_led("green")
            else:
                print(f"[{pct}] Only Siren – signal stays GREEN")
        elif class_id == CLASS_AMBULANCE_TRAFFIC:
            if self.current_led != "green":
                print(f"\n[{pct}] Ambulance detected for 2s -> Current signal (ambulance) changed to GREEN")
                self.current_led = "green"
                set_led("green")
            else:
                print(f"[{pct}] Ambulance – signal stays GREEN")

def main():
    print("=" * 60)
    print("  Ambulance Sound Detector + LED Traffic Light")
    print("=" * 60)

    # Show microphones
    print("\nAVAILABLE MICROPHONES:")
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            print(f"  [{i}] {dev['name']}")

    choice = input("\n  Type microphone number and press Enter: ").strip()
    mic_index = int(choice)
    print(f"\nUsing microphone [{mic_index}]: {sd.query_devices(mic_index)['name']}")

    # Load model
    print("\nLoading model...")
    interp, inp, out = load_model()
    print("Model loaded!")

    # Setup
    setup_gpio()
    controller = SignalController()

    # Live audio buffer
    samples_per_window = int(SAMPLE_RATE * WINDOW_SEC)
    audio_buffer   = np.zeros(samples_per_window, dtype=np.float32)
    buffer_lock    = threading.Lock()
    new_data_event = threading.Event()

    def audio_callback(indata, frames, time_info, status):
        # This runs automatically every time mic captures new audio
        with buffer_lock:
            audio_buffer[:] = indata[:samples_per_window, 0]
        new_data_event.set()

    print("\n" + "=" * 60)
    print("  LIVE Microphone is now OPEN and LISTENING...")
    print("  Press Ctrl+C to stop")
    print("=" * 60 + "\n")
    print(f"{'CLASS':<25} {'CONF':>7}   STATUS")
    print("-" * 60)

    # Open LIVE microphone stream - runs continuously
    with sd.InputStream(
        device=mic_index,
        channels=1,
        samplerate=SAMPLE_RATE,
        blocksize=samples_per_window,
        dtype="float32",
        callback=audio_callback
    ):
        print("Microphone stream ACTIVE!\n")
        try:
            while True:
                # Wait for new audio chunk from microphone
                new_data_event.wait()
                new_data_event.clear()

                # Copy latest audio
                with buffer_lock:
                    audio_chunk = audio_buffer.copy()

                # Run detection
                class_id, conf = run_inference(interp, inp, out, audio_chunk)
                label = LABEL_NAMES.get(class_id, f"Class {class_id}")

                if conf >= CONFIDENCE_TH:
                    print(f"{label:<25} {conf*100:>6.1f}%   counting toward 2s...")
                    controller.update(class_id, conf)
                else:
                    print(f"{label:<25} {conf*100:>6.1f}%   low confidence – ignored")
                    controller.update(None, conf)

        except KeyboardInterrupt:
            print("\n\nStopped by user.")
        finally:
            cleanup_gpio()
            print("Done.")

if __name__ == "__main__":
    main()
