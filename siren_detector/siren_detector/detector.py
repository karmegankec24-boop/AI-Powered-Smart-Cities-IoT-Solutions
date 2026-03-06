import tensorflow as tf
import numpy as np
import sounddevice as sd
from datetime import datetime
import sys
import os

MODEL_FILE = "soundclassifier_with_metadata.tflite"

CLASS_NAMES = {
    0: "Background Noise",
    1: "Siren only",
    2: "Traffic noise+siren"
}

SAMPLE_RATE = 16000
AUDIO_SAMPLES = 44032

print("\n" + "="*80)
print("🚨 AUDIO SIREN DETECTOR")
print("="*80)

if not os.path.exists(MODEL_FILE):
    print(f"❌ ERROR: {MODEL_FILE} not found!")
    sys.exit(1)

print("✅ Model file found!")

try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_FILE)
    interpreter.allocate_tensors()
    print("✅ Model loaded!")
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"✅ Ready!")
print(f"   Input shape: {input_details[0]['shape']}")

print("\n" + "="*80)
print("🎤 LISTENING FOR SIRENS...")
print("="*80)
print("⏹️  Press Ctrl+C to stop\n")

total_detections = 0

try:
    while True:
        audio = sd.rec(AUDIO_SAMPLES, samplerate=SAMPLE_RATE, channels=1)
        sd.wait()
        audio = audio.flatten()
        
        audio_float = np.array(audio).astype(np.float32)
        audio_float = audio_float.reshape(input_details[0]['shape'])
        
        interpreter.set_tensor(input_details[0]['index'], audio_float)
        interpreter.invoke()
        
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
        
        best_class = np.argmax(predictions)
        best_conf = predictions[best_class]
        
        class_name = CLASS_NAMES.get(best_class, f"Class {best_class}")
        
        print(f"{class_name:25s} | Confidence: {best_conf*100:5.1f}%")
        
        if best_class in [1, 2] and best_conf > 0.95:
            total_detections += 1
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            print("\n" + "="*80)
            print(f"🚨 SIREN DETECTED! - {class_name}")
            print(f"   Confidence: {best_conf*100:.1f}%")
            print(f"   Time: {timestamp}")
            print(f"   Detection #{total_detections}")
            print("="*80 + "\n")
            
            try:
                import winsound
                winsound.Beep(1000, 500)
            except:
                pass
            
            with open("detections.log", "a") as f:
                f.write(f"{timestamp} | {class_name} ({best_conf*100:.1f}%)\n")

except KeyboardInterrupt:
    print(f"\n\n⏹️  STOPPED - Total detections: {total_detections}\n")