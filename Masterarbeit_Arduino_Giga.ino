/*
  SmartHelmet (Arduino Giga R1 WiFi) – Wire2 @ 0x69
  Sensoren: Adafruit_MPU6050 (I2C), Pulse (ADC), Grove GSR (ADC)
  Aktoren: 3 LEDs (Ampel)
  Log: 1x pro Sekunde CSV über Serial (115200 Baud)

  CSV-Format:
  millis,bpm,gyro_yaw_rate_dps,pitch_deg,yaw_deg,head_status,gsr_raw,gsr_level,led_state
*/

#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <math.h>

// ========================== Pin-Definitionen ==========================
// LEDs (Ampel-Ausgabe)
const int PIN_LED_GRUEN  = 2;
const int PIN_LED_GELB   = 3;
const int PIN_LED_ROT    = 4;

// Analogeingänge für Sensoren
const int PIN_PULSE = A0;   // Pulssensor
const int PIN_GSR   = A1;   // Grove GSR-Sensor (Hautleitwert)

// Hilfsfunktion: alle drei LEDs gleichzeitig setzen
void setLED(bool g, bool y, bool r) {
  digitalWrite(PIN_LED_GRUEN, g ? HIGH : LOW);
  digitalWrite(PIN_LED_GELB,  y ? HIGH : LOW);
  digitalWrite(PIN_LED_ROT,   r ? HIGH : LOW);
}

// ========================== Zustände/Enums ==========================
// Schweregradstufen für die Ampel-Logik
enum Severity { SCHW_NORMAL = 0, SCHW_MITTEL = 1, SCHW_ERNST = 2 };
Severity gsrToSeverity(int gsrRaw);

// Globale Zustände
volatile int currentBPM = 0;   // berechnete Herzfrequenz
int baselineBPM = 0;           // Ruhe-BPM nach Kalibrierung
Severity sevPulse = SCHW_NORMAL;
Severity sevGSR   = SCHW_NORMAL;
Severity sevGyro  = SCHW_NORMAL;

// IMU (MPU6050)
Adafruit_MPU6050 mpu;
bool imuReady = false;
float yaw_deg   = 0.0f;        // integrierter Yaw-Winkel (Drehung links/rechts)
float pitch_deg = 0.0f;        // Neigung vor/zurück
unsigned long lastIMU_ms = 0;

// Kopfstatus (links / rechts / center)
String headStatus = "center";
unsigned long headLastChangeMs = 0;

// Parameter für Yaw-/Head-Erkennung
char  YAW_AXIS = 'y';           // Achse für Yaw-Berechnung
float YAW_RATE_DEADZONE = 3.0f; // Rauschunterdrückung (deg/s)
float YAW_DECAY_TAU_S   = 8.0f; // Zerfall für Yaw-Integration
float GIER_L            = -8.0f; // Schwellwert für "links"
float GIER_R            =  8.0f; // Schwellwert für "rechts"
const float ABLENKUNGS_ZEIT_S = 1.0f; // wie lange man abgelenkt sein muss

// LED-Zustand
unsigned long lastLEDUpdateMs = 0;
const unsigned long LED_AKT_INTERVAL_MS = 2000; // LED-Update alle 2s
String ledStateStr = "UNKNOWN";

// Logging (1x pro Sekunde)
unsigned long lastLogSec = 0;

// ========================== Puls-Detektion ==========================
// Sampling-Rate für Pulssensor
const unsigned long PULSE_SAMPLE_INTERVAL_MS = 10; // 100 Hz

// Gleitender Mittelwert
const int SMOOTH_N = 8;
int smoothBuf[SMOOTH_N];
int smoothIdx = 0;
long smoothSum = 0;

// Variablen für Peak-Erkennung
int dynThresh = 0;              // dynamische Schwelle
bool above = false;             // aktuell über Schwelle?
unsigned long lastPeakMs = 0;   // Zeitstempel letzter Peak
const unsigned long REFRACT_MS = 300;  // Refraktärzeit (keine Doppelzählung)

// RR-Intervall-Puffer (zur Glättung von BPM)
const int RR_MAX = 8;
unsigned long rrBuf[RR_MAX];
int rrCount = 0;

// Initialisierung Puls-Array
void pulseInit() {
  for (int i = 0; i < SMOOTH_N; i++) smoothBuf[i] = 0;
  smoothIdx = 0; smoothSum = 0;
  dynThresh = 2048; // Startwert bei 12-bit ADC
  rrCount = 0;
  lastPeakMs = 0;
}

// Update: liest Sensor, erkennt Peaks und berechnet BPM
void pulseUpdate() {
  unsigned long now = millis();
  if (now - lastPulseSampleMs < PULSE_SAMPLE_INTERVAL_MS) return;
  lastPulseSampleMs = now;

  int raw = analogRead(PIN_PULSE);

  // Glättung durch gleitenden Mittelwert
  smoothSum -= smoothBuf[smoothIdx];
  smoothBuf[smoothIdx] = raw;
  smoothSum += raw;
  smoothIdx = (smoothIdx + 1) % SMOOTH_N;
  int smoothed = smoothSum / SMOOTH_N;

  // Dynamische Schwelle nachführen
  dynThresh = (dynThresh * 99 + smoothed) / 100;

  // Peak-Detektion
  if (!above && smoothed > dynThresh + 20) {
    if (now - lastPeakMs > REFRACT_MS) {
      // RR-Intervall speichern
      if (lastPeakMs > 0 && rrCount < RR_MAX) {
        rrBuf[rrCount++] = now - lastPeakMs;
      } else if (lastPeakMs > 0) {
        for (int i = 1; i < RR_MAX; i++) rrBuf[i - 1] = rrBuf[i];
        rrBuf[RR_MAX - 1] = now - lastPeakMs;
      }
      lastPeakMs = now;

      // BPM aus Mittelwert der RR-Intervalle
      if (rrCount > 0) {
        unsigned long sum = 0;
        for (int i = 0; i < rrCount; i++) sum += rrBuf[i];
        float avgRR = sum / (float)rrCount; // ms
        if (avgRR > 0) currentBPM = (int)round(60000.0f / avgRR);
      }
    }
    above = true;
  } else if (above && smoothed < dynThresh) {
    above = false;
  }
}

// ========================== IMU (MPU6050) ==========================
// Initialisiert MPU auf Wire2 @ 0x69
void imuInit() {
  Wire2.begin();
  imuReady = mpu.begin(0x69, &Wire2);

  if (imuReady) {
    Serial.println(F("[OK] MPU6050 auf Wire2 @ 0x69 erkannt."));
    mpu.setAccelerometerRange(MPU6050_RANGE_4_G);
    mpu.setGyroRange(MPU6050_RANGE_250_DEG);
    mpu.setFilterBandwidth(MPU6050_BAND_44_HZ);
  } else {
    Serial.println(F("[WARN] MPU6050 nicht gefunden – Verkabelung/Adresse prüfen."));
  }
  lastIMU_ms = millis();
}

// Liest Gyro/Acc-Daten, berechnet yaw_deg/pitch_deg und Kopfstatus
void imuUpdate() {
  if (!imuReady) return;

  unsigned long now = millis();
  float dt = (now - lastIMU_ms) / 1000.0f;
  if (dt <= 0) dt = 0.001f;
  lastIMU_ms = now;

  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  // Gyro in deg/s
  float gx = g.gyro.x * 180.0f / PI;
  float gy = g.gyro.y * 180.0f / PI;
  float gz = g.gyro.z * 180.0f / PI;

  float yaw_rate = (YAW_AXIS=='x' ? gx : (YAW_AXIS=='y' ? gy : gz));
  if (fabsf(yaw_rate) < YAW_RATE_DEADZONE) yaw_rate = 0.0f;

  // Integration + Zerfall
  yaw_deg += (-yaw_rate) * dt;
  yaw_deg *= expf(-dt / YAW_DECAY_TAU_S);

  // Pitch aus Beschleunigung berechnen
  const float g_const = 9.80665f;
  float axg = a.acceleration.x / g_const;
  float ayg = a.acceleration.y / g_const;
  float azg = a.acceleration.z / g_const;
  pitch_deg = atan2f(-axg, sqrtf(ayg * ayg + azg * azg)) * 180.0f / PI;

  // Kopfstatus wechseln bei Schwellwertüberschreitung
  String newStatus = headStatus;
  if (yaw_deg < GIER_L && headStatus != "links") {
    newStatus = "links"; headLastChangeMs = now;
  } else if (yaw_deg > GIER_R && headStatus != "rechts") { // geändert
    newStatus = "rechts"; headLastChangeMs = now;          // geändert
  } else if (yaw_deg > GIER_L && yaw_deg < GIER_R && headStatus != "center") {
    newStatus = "center"; headLastChangeMs = now;
  }
  if (newStatus != headStatus) headStatus = newStatus;

  // Schweregrad abhängig von Neigung & Ablenkung
  bool abgelenkt = ((headStatus == "links") || (headStatus == "rechts")) &&
                   ((now - headLastChangeMs) > (unsigned long)(ABLENKUNGS_ZEIT_S * 1000.0f));

  // Pitch-Schwelle jetzt ±20° statt 30°
  bool starkGeneigt = (pitch_deg > 20.0f) || (pitch_deg < -20.0f);

  if (starkGeneigt && abgelenkt)      sevGyro = SCHW_ERNST;
  else if (starkGeneigt || abgelenkt) sevGyro = SCHW_MITTEL;
  else                                sevGyro = SCHW_NORMAL;
}

// ========================== GSR ==========================
// Rohwert lesen (12 Bit)
int readGSR() {
  return analogRead(PIN_GSR);
}

// Heuristische Schwellenwerte (anpassen!)
Severity gsrToSeverity(int gsrRaw) {
  if (gsrRaw > 2200) return SCHW_ERNST;
  if (gsrRaw > 1300) return SCHW_MITTEL;
  return SCHW_NORMAL;
}

// ========================== LED-Logik ==========================
// Setzt Ampelzustand abhängig von Puls/GSR/IMU
void updateLEDs() {
  unsigned long now = millis();
  if (now - lastLEDUpdateMs < LED_AKT_INTERVAL_MS) return;
  lastLEDUpdateMs = now;

  // Pulsabweichung von Baseline
  if (baselineBPM <= 0 || currentBPM == 0) {
    sevPulse = SCHW_NORMAL;
  } else if (currentBPM < baselineBPM - 5) {
    sevPulse = SCHW_ERNST;
  } else if (currentBPM < baselineBPM - 2) {
    sevPulse = SCHW_MITTEL;
  } else {
    sevPulse = SCHW_NORMAL;
  }

  sevGSR = gsrToSeverity(readGSR());

  int severeCnt = (sevPulse == SCHW_ERNST) + (sevGSR == SCHW_ERNST) + (sevGyro == SCHW_ERNST);
  int mediumCnt = (sevPulse == SCHW_MITTEL) + (sevGSR == SCHW_MITTEL) + (sevGyro == SCHW_MITTEL);

  if (severeCnt >= 2) {
    setLED(false, false, true);  ledStateStr = "RED";
  } else if (severeCnt == 1 && mediumCnt >= 1) {
    digitalWrite(PIN_LED_ROT, HIGH);
    digitalWrite(PIN_LED_GELB, HIGH);
    digitalWrite(PIN_LED_GRUEN, LOW);
    ledStateStr = "RED+YELLOW";
  } else if (severeCnt == 1) {
    setLED(false, true, false);  ledStateStr = "YELLOW";
  } else if (mediumCnt >= 2) {
    digitalWrite(PIN_LED_GRUEN, HIGH);
    digitalWrite(PIN_LED_GELB, HIGH);
    digitalWrite(PIN_LED_ROT, LOW);
    ledStateStr = "YELLOW+GREEN";
  } else {
    setLED(true, false, false);  ledStateStr = "GREEN";
  }
}

// ========================== Kalibrierung ==========================
// Ermittelt Ruhepuls (Baseline) in ~10 s
void calibratePulse(unsigned long durationMs = 10000) {
  Serial.println(F("Kalibriere Puls (~10 s) – bitte ruhig bleiben..."));
  unsigned long start = millis();
  int acc = 0;
  int count = 0;
  int lastBPMLocal = -1;

  while (millis() - start < durationMs) {
    pulseUpdate();
    if (currentBPM > 0 && currentBPM != lastBPMLocal) {
      acc += currentBPM;
      count++;
      lastBPMLocal = currentBPM;
    }
    delay(1);
  }
  baselineBPM = (count > 0) ? (acc / count) : currentBPM;
  Serial.print(F("Baseline BPM = ")); Serial.println(baselineBPM);
}

// ========================== Setup ==========================
void setup() {
  Serial.begin(115200);
  while (!Serial) { ; }

  pinMode(PIN_LED_GRUEN, OUTPUT);
  pinMode(PIN_LED_GELB,  OUTPUT);
  pinMode(PIN_LED_ROT,   OUTPUT);

  // LED-Selbsttest
  setLED(true, true, true); delay(800); setLED(false, false, false);

  // 12-Bit ADC-Auflösung
  analogReadResolution(12);

  pulseInit();
  imuInit();
  calibratePulse(10000);

  // CSV-Header ausgeben
  Serial.println(F("millis,bpm,gyro_yaw_rate_dps,pitch_deg,yaw_deg,head_status,gsr_raw,gsr_level,led_state"));
}

// ========================== Loop ==========================
void loop() {
  pulseUpdate();
  imuUpdate();
  updateLEDs();

  // CSV-Ausgabe alle 1 s
  unsigned long sec = millis() / 1000;
  if (sec != lastLogSec) {
    lastLogSec = sec;

    float yaw_rate_out = 0.0f;
    if (imuReady) {
      sensors_event_t a, g, t;
      mpu.getEvent(&a, &g, &t);
      float gx = g.gyro.x * 180.0f / PI;
      float gy = g.gyro.y * 180.0f / PI;
      float gz = g.gyro.z * 180.0f / PI;
      yaw_rate_out = (YAW_AXIS=='x' ? gx : (YAW_AXIS=='y' ? gy : gz));
    }

    int gsrRaw = readGSR();
    Severity gsrLevel = gsrToSeverity(gsrRaw);

    // CSV-Zeile ausgeben
    Serial.print(millis());      Serial.print(',');
    Serial.print(currentBPM);    Serial.print(',');
    Serial.print(yaw_rate_out,2);Serial.print(',');
    Serial.print(pitch_deg, 2);  Serial.print(',');
    Serial.print(yaw_deg,   2);  Serial.print(',');
    Serial.print(headStatus);    Serial.print(',');
    Serial.print(gsrRaw);        Serial.print(',');
    Serial.print((int)gsrLevel); Serial.print(',');
    Serial.println(ledStateStr);
  }
}
