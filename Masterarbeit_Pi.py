#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Clean logging build:
- Loggt 1×/Sekunde in CSV (USB-Laufwerk wird automatisch gesucht).
- Spalten: timestamp, bpm, ear, lip_distance, gyro_z_dps, pitch_deg, gier_winkel,
           kopf_status, blink_rate_per_min, blink_events_sec, yawn_events_sec, yawn_total, led_state
- Blinzel-Erkennung: kurze EAR-Dips (0.05–0.8 s) unterhalb Schwellwert.
- Gähn-Erkennung: Flanke bei lip_distance > YAWN_THRESH; Zählung pro Sekunde.
- Benötigt: Picamera2 + OpenCV + dlib (68 Landmarks) + ADS1015 (Puls) + MPU6050 (Gyro/Accel) + GPIO-LEDs.
"""

# -------------------------- Imports --------------------------
from scipy.spatial import distance as dist
from picamera2 import Picamera2
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import os
import RPi.GPIO as GPIO
from mpu6050 import mpu6050
import math
import threading
from adafruit_ads1x15.ads1015 import ADS1015
from adafruit_ads1x15.analog_in import AnalogIn
import board, busio
import csv

# -------------------------- Globale Konfiguration --------------------------

# Gähn-Fenster für "Schweregrad"-Klassifikation (die reine Ereigniszählung läuft separat pro Sekunde)
GAEHN_FENSTER = 300           # Sekunden (5 Minuten)
ERNSTE_GAEHNER = 5            # >5 Gähner im Fenster -> "ernst"
MITTLERE_GAEHNER_MIN = 3      # 3-5 -> "mittel"

# Schweregrad-Codes (Ampellogik)
SCHW_NORMAL = 0
SCHW_MITTEL = 1
SCHW_ERNST  = 2

gaehn_schweregrad = SCHW_NORMAL
augen_schweregrad = SCHW_NORMAL
augen_geschlossen_schweregrad = SCHW_NORMAL  # PERCLOS-ähnliche Minutenauswertung

# -------------------------- ADS1015 / Puls --------------------------
# I²C-Bus initialisieren und ADS1015 als ADC für Pulssignal verwenden
i2c = busio.I2C(board.SCL, board.SDA)
ads = ADS1015(i2c)
ads.gain = 1                 # Verstärkung (abhängig vom Signalpegel)
chan = AnalogIn(ads, 0)      # Puls an A0
puls_schweregrad = 0
basis_bpm = None             # Kalibrierter Basis-Puls
puls_sperre = threading.Lock()
current_bpm = 0              # BPM wird vom Pulsthread hier aktualisiert

# -------------------------- Kopf/Gyro --------------------------
# Status der Kopforientierung (aus Yaw/Gier abgeleitet)
kopf_status = "center"        # "links", "recht", "center"
kopf_letzte_aenderung = 0
DREH_SCHWELLE = 50            # (nicht direkt genutzt; GIER_* Grenzen darunter)
ABLENKUNGS_ZEIT = 1.0         # s, ab wann Seitblick als "Ablenkung" gilt

vorherige_zeit = time.time()

# -------------------------- GPIO / LEDs --------------------------
GPIO.setmode(GPIO.BCM)
GRUEN_PIN = 17
GELB_PIN  = 27
ROT_PIN   = 22
GPIO.setup(GRUEN_PIN, GPIO.OUT)
GPIO.setup(GELB_PIN,  GPIO.OUT)
GPIO.setup(ROT_PIN,   GPIO.OUT)

def setze_leds(g=False, y=False, r=False):
    """Schaltet die drei Status-LEDs entsprechend der Ampellogik."""
    GPIO.output(GRUEN_PIN, g)
    GPIO.output(GELB_PIN,  y)
    GPIO.output(ROT_PIN,   r)

# -------------------------- MPU6050 --------------------------
sensor = mpu6050(0x68)
GYRO_SCHWELLE = 50            # Reserve 
gier_winkel = 0.0             # Integrierter Z-Gyro (Gier)
vorherige_zeit = time.time()
GIER_LINKS_SCHWELLE  = -20    # Schwellwerte für "links/recht/center"
GIER_RECHTS_SCHWELLE =  20
NEIGUNG_DAUER = 2.0           # s, wie lange Pitch >20° anliegen muss, um "Kopf unten" zu werten
neigung_start_zeit = None
letzte_neigung = 0.0
kopf_unten_aktiv = False
ablenkung_aktiv  = False

# -------------------------- LED-Refresh-Rate --------------------------
letzte_led_aktualisierung = -9999.0
LED_AKT_INTERVAL = 2.0        # LEDs nur alle 2 s updaten (Flackern vermeiden)
led_state_str = "UNKNOWN"     # Wird auch geloggt

# -------------------------- Müdigkeit/Gähnen/Augen --------------------------
letzter_muedigkeitszeitpunkt = 0.0
letzter_gaehnzeitpunkt = 0.0
schlaefrigkeit_startzeit = None

# EAR-Threshold (Eye Aspect Ratio) für "Auge geschlossen"
auge_AR_THRESH = 0.3
# Schwelle für Mundöffnung (Pixelabstand Ober-/Unterlippe)
YAWN_THRESH = 25.0

# Timer/Akkumulatoren für "Augen geschlossen" (PERCLOS-ähnlich über 60 s)
augen_geschlossen_beginn = None     # Zeitpunkt Start eines geschlossenen Zustands
augen_geschlossen_gesamt = 0.0      # Summe geschlossener Zeit in der aktuellen Minute
augen_Minute_beginn = None          # Startzeitpunkt des 60-s-Fensters

# Gähnzählung (für Fenster-basierte Schweregrad-Einstufung)
gaehn_zeitpunkte = []
gaehn_ongoing = False               # true während eines erkannten Gähnens (verhindert Doppeltzählung)

# Blinzelerkennung
blink_in_progress = False
blink_start_time = None
blink_events_this_sec = 0
blink_timestamps = []               # Zeitstempel der letzten 60 s, um Rate/min zu berechnen

# -------------------------- Logging (CSV) --------------------------
def _find_usb_dir():
    """Sucht ein beschreibbares Verzeichnis (bevorzugt USB-Mounts). Fallback: aktuelles Verzeichnis."""
    candidates = ["/media/pi", "/media", "/mnt/usb", "/mnt", "/home/pi"]
    for base in candidates:
        if not os.path.isdir(base):
            continue
        try:
            for entry in sorted(os.listdir(base)):
                full = os.path.join(base, entry)
                if os.path.isdir(full):
                    try:
                        # Schreibprobe
                        testfile = os.path.join(full, ".usb_write_test")
                        with open(testfile, "w") as f:
                            f.write("ok")
                        os.remove(testfile)
                        return full
                    except Exception:
                        continue
        except Exception:
            continue
    return "."

# Eigener fester Pfad falls vorhanden, sonst automatisch finden
LOG_DIR = "/media/benni/31 GB Volume" if os.path.isdir("/media/benni/31 GB Volume") else _find_usb_dir()  # <— ggf. anpassen
print(f"[LOG] LOG_DIR: {LOG_DIR}")
LOG_FILE = None
_csv_writer = None
csv_file = None
last_log_sec = int(time.time())

# -------------------------- Hilfsfunktionen --------------------------
def alarm(msg):
    """Einfache Alarm-Ausgabe (kann später z. B. Tonausgabe triggern)."""
    print(msg)

def auge_aspect_ratio(auge):
    """Berechnet EAR (Eye Aspect Ratio) aus 6 Landmark-Punkten eines Auges."""
    A = dist.euclidean(auge[1], auge[5])
    B = dist.euclidean(auge[2], auge[4])
    C = dist.euclidean(auge[0], auge[3])
    ear = (A + B) / (2.0 * C)
    return ear

def final_ear(shape):
    """EAR beider Augen berechnen und mitteln. Liefert (EAR, linkes_auge, rechtes_auge)."""
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    linkes_auge  = shape[lStart:lEnd]
    rechtes_auge = shape[rStart:rEnd]
    linksEAR = auge_aspect_ratio(linkes_auge)
    rechtEAR = auge_aspect_ratio(rechtes_auge)
    ear = (linksEAR + rechtEAR) / 2.0
    return (ear, linkes_auge, rechtes_auge)

def lip_distance(shape):
    """Vertikale Lippen-Distanz (als Proxy für Mundöffnung)."""
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))
    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))
    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)
    distance = abs(top_mean[1] - low_mean[1])
    return distance

def berechne_neigung(sensor, baseline_vektor):
    """Berechnet den Pitch (Vor-/Rückneigung) relativ zum Baseline-Vektor (Gyro-Kalibrierlage)."""
    try:
        beschl = sensor.get_accel_data()
        ax, ay, az = beschl['x'], beschl['y'], beschl['z']
        vec = np.array([ax, ay, az], dtype=np.float64)
        vec /= np.linalg.norm(vec)

        # Achsen aufspannen (seitlich/vorwärts) relativ zur Baseline
        seitliche_achse = np.cross(baseline_vektor, [0, 0, 1])
        seitliche_achse /= np.linalg.norm(seitliche_achse)

        vorwaerts_achse = np.cross(seitliche_achse, baseline_vektor)
        vorwaerts_achse /= np.linalg.norm(vorwaerts_achse)

        # Signierter Winkel entlang der Vorwärtsachse
        dot = np.dot(vec, vorwaerts_achse)
        neigungswinkel = math.degrees(math.asin(dot))
        return neigungswinkel
    except Exception as e:
        print(f"[WARNUNG] MPU6050-Lesefehler: {e}")
        return None

def hole_bpm(dauer_sec=10, abtastrate_hz=100):
    """
    Schätzt BPM aus ADC-Signal:
    - Abtasten über dauer_sec
    - Glätten (Moving Average)
    - Dynamische Schwelle (90. Perzentil)
    - Peak-Detektion mit Mindestabstand (0.5 s -> max ~120 BPM)
    """
    anzahl = dauer_sec * abtastrate_hz
    verz = 1.0 / abtastrate_hz
    signal = []
    zeitstempel = []
    for _ in range(anzahl):
        wert = chan.value
        signal.append(wert)
        zeitstempel.append(time.time())
        time.sleep(verz)

    signal = np.array(signal, dtype=np.float64)
    zeitstempel = np.array(zeitstempel)

    # DC-Offset entfernen + glätten
    signal -= np.mean(signal)
    signal = np.convolve(signal, np.ones(5)/5, mode='same')

    # Peaks suchen
    schwelle = np.percentile(signal, 90)
    spitzen_indices = np.where(
        (signal[1:-1] > schwelle) &
        (signal[1:-1] > signal[:-2]) &
        (signal[1:-1] > signal[2:])
    )[0] + 1

    # Mindestabstand zwischen Peaks (0.5 s)
    min_intervall = 0.5
    gef_spitzen_zeiten = []
    for idx in spitzen_indices:
        t = zeitstempel[idx]
        if not gef_spitzen_zeiten or (t - gef_spitzen_zeiten[-1]) > min_intervall:
            gef_spitzen_zeiten.append(t)

    if len(gef_spitzen_zeiten) < 2:
        print("Keine gültigen Herzschläge erkannt.")
        return 0

    intervals = np.diff(gef_spitzen_zeiten)
    bpm = 60.0 / np.mean(intervals)
    return round(bpm)

def kalibrierung(sensor, dauer_sec=60, abtastrate_hz=100):
    """
    Kombinierte Kalibrierung:
    - Puls-Basis (BPM) über Dauer mitteln (Peak-Ansatz wie oben)
    - Baseline-Vektor für Pitch-Berechnung (Mittel der Beschleuniger-Stichproben)
    Hinweis: Erfordert ruhige Kopfhaltung in der Kalibrierphase.
    """
    print(f"\n\nKalibrierung der Herzfrequenz und MPU6050... \nHalte den Kopf für {dauer_sec} Sekunden möglichst ruhig.")
    anzahl = dauer_sec * abtastrate_hz
    verz = 1.0 / abtastrate_hz
    signal = []
    zeitstempel = []
    stichproben = []

    for _ in range(anzahl):
        # Puls
        wert = chan.value
        signal.append(wert)
        zeitstempel.append(time.time())

        # Accel
        beschl = sensor.get_accel_data()
        ax, ay, az = beschl['x'], beschl['y'], beschl['z']
        stichproben.append((ax, ay, az))

        time.sleep(verz)

    # Puls-Basis bestimmen (wie in hole_bpm)
    signal = np.array(signal, dtype=np.float64)
    zeitstempel = np.array(zeitstempel)
    signal -= np.mean(signal)
    signal = np.convolve(signal, np.ones(5)/5, mode='same')

    schwelle = np.percentile(signal, 90)
    spitzen_indices = np.where(
        (signal[1:-1] > schwelle) &
        (signal[1:-1] > signal[:-2]) &
        (signal[1:-1] > signal[2:])
    )[0] + 1

    min_intervall = 0.5
    gef_spitzen_zeiten = []
    for idx in spitzen_indices:
        t = zeitstempel[idx]
        if not gef_spitzen_zeiten or (t - gef_spitzen_zeiten[-1]) > min_intervall:
            gef_spitzen_zeiten.append(t)

    if len(gef_spitzen_zeiten) < 2:
        print("Keine gültigen Herzschläge erkannt.")
        return 0

    intervals = np.diff(gef_spitzen_zeiten)
    bpm = 60.0 / np.mean(intervals)

    # Baseline-Vektor für Pitch
    durchschnitt = np.mean(stichproben, axis=0)
    baseline_vektor = durchschnitt / np.linalg.norm(durchschnitt)
    print(f"Basis-Vorwärtsvektor: {baseline_vektor}")

    return round(bpm), baseline_vektor

def puls_monitor_thread():
    """
    Hintergrund-Thread:
    - Misst alle 10 s BPM
    - Vergleicht mit Basis-BPM und setzt Schweregrad (normal/mittel/ernst)
    """
    global puls_schweregrad, basis_bpm, current_bpm
    print("Initialisierung des Pulsmonitors...")
    while True:
        bpm_val = hole_bpm(dauer_sec=10)
        print(f"Aktueller BPM: {bpm_val}")
        current_bpm = bpm_val
        with puls_sperre:
            if basis_bpm is None or bpm_val == 0:
                puls_schweregrad = SCHW_NORMAL
            elif bpm_val < basis_bpm - 5:
                print("BPM mindestens 5 niedriger als Kalibrierwert -> ernst")
                puls_schweregrad = SCHW_ERNST
            elif bpm_val < basis_bpm - 2:
                print("BPM mindestens 2 niedriger als Kalibrierwert -> mittel")
                puls_schweregrad = SCHW_MITTEL
            else:
                puls_schweregrad = SCHW_NORMAL

# -------------------------- Kommandozeilen-Argumente --------------------------
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0, help="Index der Webcam im System (nicht genutzt für PiCam)")
args = vars(ap.parse_args())

# Schwellen/Flags für Augen/Gähnen (einige Variablen bleiben aus Kompat. erhalten)
auge_AR_CONSEC_FRAMES = 30
alarm_status  = False
alarm_status2 = False
COUNTER = 0

# -------------------------- Modelle/Kamera starten --------------------------
print("-> Laden des Prädiktors und Detektors...")
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")       # Haar-Cascade für Gesichtsbox
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')     # 68-Punkte-Landmarks

print("-> Videostream starten")
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()
time.sleep(1.0)

# LED Selbsttest
setze_leds(g=True, y=True, r=True)
time.sleep(1.0)
setze_leds(g=False, y=False, r=False)

# Kalibrierung (60 s): Basis-BPM + Baseline-Vektor
basis_bpm, baseline_vektor = kalibrierung(sensor, dauer_sec=60)
print(f"Kalibrierungswert: {basis_bpm} BPM")

# Pulsthread starten
pulse_thread = threading.Thread(target=puls_monitor_thread, daemon=True)
pulse_thread.start()

# -------------------------- CSV öffnen --------------------------
def open_csv():
    """Öffnet/erstellt CSV im Unterordner 'smarthelmet_logs' inkl. Headerzeile."""
    global LOG_FILE, _csv_writer, csv_file
    try:
        ts = time.strftime("%Y%m%d-%H%M%S")
        out_dir = os.path.join(LOG_DIR, "smarthelmet_logs")
        # Schreibprobe im LOG_DIR
        try:
            _test = os.path.join(LOG_DIR, ".usb_write_test")
            with open(_test, "w") as _f:
                _f.write("ok")
            os.remove(_test)
        except Exception as _e:
            print(f"[LOG] WARN: Kein Schreibzugriff auf {LOG_DIR}: {_e}. Fallback: aktuelles Verzeichnis")
            out_dir = os.path.join(".", "smarthelmet_logs")

        os.makedirs(out_dir, exist_ok=True)
        LOG_FILE = os.path.join(out_dir, f"sensordata_{ts}.csv")
        csv_file = open(LOG_FILE, "a", newline="")
        _csv_writer = csv.writer(csv_file)
        _csv_writer.writerow([
            "timestamp","bpm","ear","lip_distance","gyro_z_dps","pitch_deg","gier_winkel",
            "kopf_status","blink_rate_per_min","blink_events_sec","yawn_events_sec","yawn_total","led_state"
        ])
        print(f"[LOG] Schreibe CSV nach: {LOG_FILE}")
    except Exception as e:
        # Fallback: aktuelles Verzeichnis
        print(f"[LOG] Konnte CSV nicht öffnen: {e}. Schreibe ins aktuelle Verzeichnis.")
        ts = time.strftime("%Y%m%d-%H%M%S")
        LOG_FILE = f"sensordata_{ts}.csv"
        csv_file = open(LOG_FILE, "a", newline="")
        _csv_writer = csv.writer(csv_file)

open_csv()
last_log_sec = int(time.time())

# Initiale Timer/Zustände
schlaefrigkeit_startzeit = None
letzte_neigung = 0.0
kopf_unten_aktiv = False
ablenkung_aktiv  = False
augen_Minute_beginn = time.time()

# -------------------------- Hauptschleife --------------------------
while True:
    # Kamera-Frame holen
    frame = picam2.capture_array()
    frame = imutils.resize(frame, width=450)
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Gesichtserkennung (Haar-Cascade liefert Bounding-Boxes)
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                      flags=cv2.CASCADE_SCALE_IMAGE)

    # Gyro lesen (Z-Achse für Gierwinkel)
    try:
        gyro_data = sensor.get_gyro_data()
        gyro_z = gyro_data['z']
    except OSError as e:
        print(f"[WARNUNG] MPU6050 Lesefehler: {e}")
        time.sleep(0.01)
        continue

    # Pitch (Vor-/Rückneigung) aus Beschleuniger + Baseline berechnen
    pitch = berechne_neigung(sensor, baseline_vektor)
    if pitch is None:
        pitch = letzte_neigung
    else:
        letzte_neigung = pitch

    # "Kopf unten" (Pitch > +20°) / "Kopf hinten" (Pitch < -20°) mit Mindestdauer
    if pitch > 20:
        if neigung_start_zeit is None:
            neigung_start_zeit = time.time()
        elif time.time() - neigung_start_zeit > NEIGUNG_DAUER:
            if not kopf_unten_aktiv:
                print(f"Kopf nach unten (Neigung={pitch:.1f}°)")
                kopf_unten_aktiv = True
    elif pitch < -20:
        if neigung_start_zeit is None:
            neigung_start_zeit = time.time()
        elif time.time() - neigung_start_zeit > NEIGUNG_DAUER:
            if not kopf_unten_aktiv:
                print(f"Kopf nach hinten (Neigung={pitch:.1f}°)")
                kopf_unten_aktiv = True
    else:
        neigung_start_zeit = None
        kopf_unten_aktiv = False

    # Gierwinkel integrieren (einfaches Dead-Reckoning + leichter Zerfall zur Driftminderung)
    now_t = time.time()
    dt = now_t - vorherige_zeit
    vorherige_zeit = now_t
    gier_winkel += (-gyro_z) * dt
    gier_winkel *= 0.98  # sanfter Zerfall

    # Diskrete Kopfstatus-Klassifikation aus Gierwinkel
    neuer_status = kopf_status
    if gier_winkel < GIER_LINKS_SCHWELLE and kopf_status != "links":
        neuer_status = "links"; kopf_letzte_aenderung = now_t
    elif gier_winkel > GIER_RECHTS_SCHWELLE and kopf_status != "recht":
        neuer_status = "recht"; kopf_letzte_aenderung = now_t
    elif GIER_LINKS_SCHWELLE < gier_winkel < GIER_RECHTS_SCHWELLE and kopf_status != "center":
        neuer_status = "center"; kopf_letzte_aenderung = now_t
    if neuer_status != kopf_status:
        kopf_status = neuer_status
        print(f"Kopf bewegt sich nach: {kopf_status.upper()}")

    # Ablenkungsdetektion: seitlicher Blick > ABLENKUNGS_ZEIT
    if kopf_status in ["links", "recht"] and (now_t - kopf_letzte_aenderung > ABLENKUNGS_ZEIT):
        if not ablenkung_aktiv:
            alarm(f"SCHAUEN {kopf_status.upper()}")
            ablenkung_aktiv = True
        cv2.putText(frame, f"SCHAUEN {kopf_status.upper()}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Gierwinkel: {gier_winkel:.2f}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
    else:
        ablenkung_aktiv = False

    # Initialwerte für diese Iteration
    ear = None
    abstand_lippen = None

    # Falls ein Gesicht erkannt wurde: Landmarking, EAR, Lippenabstand, Overlays, Events …
    for (x, y, w, h) in rects:
        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        ear_data = final_ear(shape)
        ear = ear_data[0]
        linkes_auge  = ear_data[1]
        rechtes_auge = ear_data[2]

        # Augen-/Lippen-Konturen zeichnen (Overlay)
        cv2.drawContours(frame, [cv2.convexHull(linkes_auge)],  -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(rechtes_auge)], -1, (0, 255, 0), 1)
        lip = shape[48:60]
        cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

        # ----------------- Augen geschlossen / PERCLOS-ähnliche Logik -----------------
        if ear < auge_AR_THRESH:   # Auge gilt als "geschlossen"
            cv2.putText(frame, "Geschlossene Augen erkannt", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if augen_geschlossen_beginn is None:
                # Startzeitpunkt dieser Schließung merken
                augen_geschlossen_beginn = now_t

            # Wenn gleichzeitig "Kopf unten", dann Müdigkeitsdauer akkumulieren
            if kopf_unten_aktiv:
                if schlaefrigkeit_startzeit is None:
                    schlaefrigkeit_startzeit = now_t
                else:
                    closed_dur = now_t - schlaefrigkeit_startzeit
                    if closed_dur > 8.0:
                        if augen_schweregrad != SCHW_ERNST:
                            print("Hohe Müdigkeitssymptome")
                        augen_schweregrad = SCHW_ERNST
                        cv2.putText(frame, "Hohe Muedigkeitssymptome", (10, 120),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    elif closed_dur > 4.0:
                        if augen_schweregrad != SCHW_MITTEL:
                            print("Mittlere Müdigkeitssymptome")
                        augen_schweregrad = SCHW_MITTEL
                        cv2.putText(frame, "Mittlere Muedigkeitssymptome", (10, 120),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            else:
                # Kopf nicht unten -> keine Schläfrigkeitsakkumulation
                schlaefrigkeit_startzeit = None
                augen_schweregrad = SCHW_NORMAL

        else:
            # Auge wieder offen -> aktuelle Schließungsdauer in Minutensumme übernehmen
            schlaefrigkeit_startzeit = None
            augen_schweregrad = SCHW_NORMAL
            if augen_geschlossen_beginn is not None:
                augen_geschlossen_gesamt += now_t - augen_geschlossen_beginn
                augen_geschlossen_beginn = None

        # Einmal pro Minute PERCLOS-ähnliche Auswertung + Reset
        if now_t - augen_Minute_beginn >= 60:
            # Falls am Fensterrand noch geschlossen, diesen Anteil hinzufügen
            if augen_geschlossen_beginn is not None:
                augen_geschlossen_gesamt += now_t - augen_geschlossen_beginn
                augen_geschlossen_beginn = now_t  # neuer Start für nächste Minute

            # Kappen auf 60 s (Sicherheitsnetz)
            if augen_geschlossen_gesamt > 60:
                augen_geschlossen_gesamt = 60

            # Ampellogik: (Gelb ab ~6 s/Min (10%), Rot ab ~30 s/Min) (50%)
            if augen_geschlossen_gesamt >= 30:
                augen_geschlossen_schweregrad = SCHW_ERNST
                print(f"Auge geschlossen: {augen_geschlossen_gesamt:.1f}s/60s -> Rot")
            elif augen_geschlossen_gesamt >= 6:
                augen_geschlossen_schweregrad = SCHW_MITTEL
                print(f"Auge geschlossen: {augen_geschlossen_gesamt:.1f}s/60s -> Gelb")
            else:
                augen_geschlossen_schweregrad = SCHW_NORMAL
                print(f"Auge geschlossen: {augen_geschlossen_gesamt:.1f}s/60s -> Gruen")

            # Fenster zurücksetzen
            augen_geschlossen_gesamt = 0.0
            augen_Minute_beginn = now_t

        # ----------------- Gähnen -----------------
        abstand_lippen = lip_distance(shape)
        if abstand_lippen > YAWN_THRESH:
            cv2.putText(frame, "Gaehnen erkannt", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if not gaehn_ongoing:
                gaehn_ongoing = True
                jetzt = time.time()
                gaehn_zeitpunkte.append(jetzt)
                # Fenster aktualisieren
                gaehn_zeitpunkte[:] = [t for t in gaehn_zeitpunkte if jetzt - t <= GAEHN_FENSTER]
                gaehn_anzahl = len(gaehn_zeitpunkte)
                # Schweregrad grob einstufen (nur Info/Overlay)
                if gaehn_anzahl > ERNSTE_GAEHNER:
                    gaehn_schweregrad = SCHW_ERNST
                    level_msg = f"Gaehn anzahl: {gaehn_anzahl}  Ernst (>5/5min)"
                elif gaehn_anzahl >= MITTLERE_GAEHNER_MIN:
                    gaehn_schweregrad = SCHW_MITTEL
                    level_msg = f"Gaehn anzahl: {gaehn_anzahl}  Mittel (3-5/5min)"
                else:
                    gaehn_schweregrad = SCHW_NORMAL
                    level_msg = f"Gaehn anzahl: {gaehn_anzahl}  Normal (<=2/5min)"
                cv2.putText(frame, level_msg, (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                print(level_msg)
                # Sekunden-Zähler für CSV (Events pro Sekunde)
                yawn_events_this_sec = yawn_events_this_sec + 1 if 'yawn_events_this_sec' in globals() else 1
                yawn_total = yawn_total + 1 if 'yawn_total' in globals() else 1
        else:
            gaehn_ongoing = False

        # ----------------- Overlays -----------------
        cv2.putText(frame, f"EAR: {ear:.2f}" if ear is not None else "EAR: n/a", (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if abstand_lippen is not None:
            cv2.putText(frame, f"Gaehnen: {abstand_lippen:.2f}", (300, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # ----------------- Blinzelerkennung -----------------
        # Kurze Schließungen (0.05–0.8 s) werden als Blink gezählt
        if ear is not None:
            if ear < auge_AR_THRESH:
                if not blink_in_progress:
                    blink_in_progress = True
                    blink_start_time = time.time()
            else:
                if blink_in_progress:
                    dauer = time.time() - blink_start_time if blink_start_time else 0.0
                    if 0.05 < dauer < 0.8:
                        blink_events_this_sec += 1
                        blink_timestamps.append(time.time())
                    blink_in_progress = False
                    blink_start_time = None

        # Nur das erste (größte) Gesicht pro Frame verarbeiten
        break

    # ----------------- LED-Ampellogik (alle 2 s) -----------------
    now_led = time.time()
    if now_led - letzte_led_aktualisierung > LED_AKT_INTERVAL:
        # Anzahl "ernst" und "mittel" über die vier Kanäle
        schwere_cnt = sum(sev == SCHW_ERNST for sev in [gaehn_schweregrad, augen_schweregrad, puls_schweregrad, augen_geschlossen_schweregrad])
        mittel_cnt  = sum(sev == SCHW_MITTEL for sev in [gaehn_schweregrad, augen_schweregrad, puls_schweregrad, augen_geschlossen_schweregrad])

        if schwere_cnt >= 2:
            setze_leds(g=False, y=False, r=True);            led_state_str = "RED"
        elif schwere_cnt == 1 and mittel_cnt >= 1:
            setze_leds(g=False, y=True, r=True);             led_state_str = "RED+YELLOW"
        elif schwere_cnt == 1:
            setze_leds(g=False, y=True, r=False);            led_state_str = "YELLOW"
        elif mittel_cnt == 2:
            setze_leds(g=True,  y=True,  r=False);           led_state_str = "YELLOW+GREEN"
        else:
            setze_leds(g=True,  y=False, r=False);           led_state_str = "GREEN"

        letzte_led_aktualisierung = now_led

    # ----------------- Blinzelrate/min berechnen (Fenster 60 s) -----------------
    now = time.time()
    blink_timestamps[:] = [t for t in blink_timestamps if now - t <= 60.0]
    blink_rate_per_min = len(blink_timestamps)

    # ----------------- CSV-Logging 1×/Sekunde -----------------
    current_sec = int(now)
    if current_sec != last_log_sec:
        try:
            if _csv_writer is not None:
                _csv_writer.writerow([
                    now,
                    current_bpm,
                    float(ear) if ear is not None else None,
                    float(abstand_lippen) if abstand_lippen is not None else None,
                    float(gyro_z) if 'gyro_z' in locals() else None,
                    float(pitch) if 'pitch' in locals() else None,
                    float(gier_winkel),
                    str(kopf_status),
                    int(blink_rate_per_min),
                    int(blink_events_this_sec),
                    int(globals().get('yawn_events_this_sec', 0)),
                    int(globals().get('yawn_total', 0)),
                    led_state_str
                ])
        except Exception as e:
            print(f"[LOG] CSV write error: {e}")

        # Sekunden-Zähler zurücksetzen (für die nächste Sekunde)
        last_log_sec = current_sec
        blink_events_this_sec = 0
        globals()['yawn_events_this_sec'] = 0

    # ----------------- Anzeige & Tastatur -----------------
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# -------------------------- Aufräumen --------------------------
cv2.destroyAllWindows()
picam2.stop()
try:
    if csv_file:
        csv_file.close()
except Exception:
    pass
GPIO.cleanup()
