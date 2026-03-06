import sys
import cv2
import numpy as np
import sounddevice as sd
import tensorflow as tf
import serial
import serial.tools.list_ports
import threading
import time

from PyQt5 import QtWidgets, QtGui, QtCore
import pyqtgraph as pg
from ultralytics import YOLO


# ------------------------------
# CONFIG
# ------------------------------

YOLO_MODEL = "best_model.pt"
TFLITE_MODEL = "../siren_detector/soundclassifier_with_metadata.tflite"

SAMPLE_RATE = 16000
AUDIO_SAMPLES = 44032

CONFIDENCE_THRESHOLD = 0.6

EMERGENCY_CLASSES = {
    "ambulance",
    "fire truck",
    "fire engine"
}

SIREN_CLASSES = {
    0: "Background Noise",
    1: "Siren only",
    2: "Traffic noise+siren"
}


# ------------------------------
# FIND ESP32
# ------------------------------

def find_esp32():

    for p in serial.tools.list_ports.comports():

        if any(x in p.description.lower() for x in
               ["cp210", "ch340", "uart", "usb serial"]):

            return p.device

    return None


# ------------------------------
# MAIN DASHBOARD
# ------------------------------

class Dashboard(QtWidgets.QMainWindow):

    def __init__(self):

        super().__init__()

        self.setWindowTitle("🚦 Ambulance Emergency Detector")
        self.setGeometry(100, 100, 1500, 900)

        # ---------------- UI LAYOUT ----------------

        main = QtWidgets.QWidget()
        self.setCentralWidget(main)

        grid = QtWidgets.QGridLayout(main)

        # CAMERA
        self.camera_label = QtWidgets.QLabel()
        self.camera_label.setFixedSize(900, 600)
        grid.addWidget(self.camera_label, 0, 0)

        # SIGNAL STATUS
        self.signal = QtWidgets.QLabel("RED")
        self.signal.setStyleSheet("font-size:60px;color:red;font-weight:bold")
        grid.addWidget(self.signal, 1, 0)

        # WAVEFORM
        self.plot = pg.PlotWidget()
        self.curve = self.plot.plot()
        grid.addWidget(self.plot, 0, 1)

        # CONFIDENCE BARS
        self.bar_siren = QtWidgets.QProgressBar()
        self.bar_noise = QtWidgets.QProgressBar()
        self.bar_mix = QtWidgets.QProgressBar()

        grid.addWidget(QtWidgets.QLabel("Siren only"), 1, 1)
        grid.addWidget(self.bar_siren, 2, 1)

        grid.addWidget(QtWidgets.QLabel("Background noise"), 3, 1)
        grid.addWidget(self.bar_noise, 4, 1)

        grid.addWidget(QtWidgets.QLabel("Traffic + siren"), 5, 1)
        grid.addWidget(self.bar_mix, 6, 1)

        # LOGS
        self.logs = QtWidgets.QTextEdit()
        grid.addWidget(self.logs, 7, 0, 1, 2)

        # ---------------- CAMERA ----------------

        self.cap = cv2.VideoCapture(0)

        # ---------------- YOLO ----------------

        self.model = YOLO(YOLO_MODEL)

        # ---------------- AUDIO MODEL ----------------

        self.interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # ---------------- ESP32 ----------------

        port = find_esp32()

        if port:

            self.esp = serial.Serial(port, 115200, timeout=1)
            self.log("ESP32 connected")

        else:

            self.esp = None
            self.log("ESP32 not detected")

        # ---------------- TIMERS ----------------

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_camera)
        self.timer.start(30)

        # AUDIO THREAD
        threading.Thread(target=self.audio_loop, daemon=True).start()

        # TRAFFIC STATE
        self.signal_state = "RED"

    # ---------------- LOG ----------------

    def log(self, msg):

        t = time.strftime("%H:%M:%S")

        self.logs.append(f"[{t}] {msg}")

    # ---------------- CAMERA UPDATE ----------------

    def update_camera(self):

        ret, frame = self.cap.read()

        if not ret:
            return

        results = self.model(frame)[0]

        emergency_detected = False

        for box in results.boxes:

            conf = float(box.conf[0])
            cls = int(box.cls[0])

            name = self.model.names[cls].lower()

            if name not in EMERGENCY_CLASSES:
                continue

            if conf > CONFIDENCE_THRESHOLD:

                emergency_detected = True

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 3)

                cv2.putText(frame, name, (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

        if emergency_detected:

            self.signal.setText("GREEN")
            self.signal.setStyleSheet("font-size:60px;color:green")

            if self.esp:
                self.esp.write(b"EMERGENCY\n")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        h, w, ch = frame.shape

        img = QtGui.QImage(frame.data, w, h, ch*w,
                           QtGui.QImage.Format_RGB888)

        self.camera_label.setPixmap(QtGui.QPixmap.fromImage(img))

    # ---------------- AUDIO LOOP ----------------

    def audio_loop(self):

        stream = sd.InputStream(callback=self.audio_callback,
                                channels=1,
                                samplerate=SAMPLE_RATE)

        with stream:
            while True:
                time.sleep(0.1)

    # ---------------- AUDIO CALLBACK ----------------

    def audio_callback(self, indata, frames, time_info, status):

        audio = indata[:,0]

        self.curve.setData(audio)

        audio = audio.astype("float32")

        audio = np.resize(audio, AUDIO_SAMPLES)

        audio = audio.reshape(self.input_details[0]["shape"])

        self.interpreter.set_tensor(self.input_details[0]["index"], audio)

        self.interpreter.invoke()

        preds = self.interpreter.get_tensor(
            self.output_details[0]["index"])[0]

        noise = int(preds[0]*100)
        siren = int(preds[1]*100)
        mix = int(preds[2]*100)

        self.bar_noise.setValue(noise)
        self.bar_siren.setValue(siren)
        self.bar_mix.setValue(mix)

        if siren > 90 or mix > 80:

            self.log("🚨 Siren detected")

            if self.signal_state == "RED":

                self.signal.setText("GREEN")
                self.signal.setStyleSheet("font-size:60px;color:green")

                if self.esp:
                    self.esp.write(b"EMERGENCY\n")


# ---------------- RUN ----------------

app = QtWidgets.QApplication(sys.argv)

window = Dashboard()

window.show()

sys.exit(app.exec_())