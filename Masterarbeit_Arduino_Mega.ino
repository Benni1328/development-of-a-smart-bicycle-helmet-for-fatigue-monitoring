/*
  SmartHelmet – Arduino MEGA 2560 Version (Wire @ 0x69)
  Sensoren: MPU6050 (I2C), Pulse (A0), Grove GSR (A1)
  Aktoren: 3 LEDs (Ampel)
  Log: 1x pro Sekunde CSV über Serial (115200 Baud)

  I2C Pins MEGA: SDA=20, SCL=21
  CSV: millis,bpm,gyro_yaw_rate_dps,pitch_deg,yaw_deg,head_status,gsr_raw,gsr_level,led_state
*/

#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <math.h>

// ---------------- Pins (MEGA) ----------------
const int PIN_LED_GRUEN  = 2;
const int PIN_LED_GELB   = 3;
const int PIN_LED_ROT    = 4;

const int PIN_PULSE = A0;   // Pulssensor (10-bit ADC: 0..1023)
const int PIN_GSR   = A1;   // GSR (10-bit ADC: 0..1023)

static inline void setLED(bool g, bool y, bool r) {
  digitalWrite(PIN_LED_GRUEN, g ? HIGH : LOW);
  digitalWrite(PIN_LED_GELB,  y ? HIGH : LOW);
  digitalWrite(PIN_LED_ROT,   r ? HIGH : LOW);
}

// ---------------- Schweregrade ----------------
enum Severity { SCHW_NORMAL = 0, SCHW_MITTEL = 1, SCHW_ERNST = 2 };
Severity gsrToSeverity(int gsrRaw);

// ---------------- Globale Zustände ----------------
volatile int currentBPM = 0;
int baselineBPM = 0;
Severity sevPulse = SCHW_NORMAL;
Severity sevGSR   = SCHW_NORMAL;
Severity sevGyro  = SCHW_NORMAL;

Adafruit_MPU6050 mpu;
bool imuReady = false;

float yaw_deg   = 0.0f;
float pitch_deg = 0.0f;
unsigned long lastIMU_ms = 0;

String headStatus = "center";
unsigned long headLastChangeMs = 0;

// ---------- Yaw/Head-Konfiguration ----------
char  YAW_AXIS = 'y';            // 'x','y' oder 'z'
float YAW_RATE_DEADZONE = 3.0f;  // deg/s
float YAW_DECAY_TAU_S   = 8.0f;  // s
float GIER_L            = -8.0f; // deg
float GIER_R            =  8.0f; // deg
const float ABLENKUNGS_ZEIT_S = 1.0f;

// ---------- LED/Log ----------
unsigned long lastLEDUpdateMs = 0;
const unsigned long LED_AKT_INTERVAL_MS = 2000;
String ledStateStr = "UNKNOWN";
unsigned long lastLogSec = 0;

// ---------------- Pulse-Detektion ----------------
const unsigned long PULSE_SAMPLE_INTERVAL_MS = 10; // ~100 Hz
unsigned long lastPulseSampleMs = 0;

const int SMOOTH_N = 8;
int  smoothBuf[SMOOTH_N];
int  smoothIdx = 0;
long smoothSum = 0;

int  dynThresh = 512;          // Start um Mittelwert (10-bit: ~512)
bool above = false;
unsigned long lastPeakMs = 0;
const unsigned long REFRACT_MS = 300;  // ms

const int RR_MAX = 8;
unsigned long rrBuf[RR_MAX];
int rrCount = 0;

void pulseInit() {
  for (int i = 0; i < SMOOTH_N; i++) smoothBuf[i] = 0;
  smoothIdx = 0; smoothSum = 0;
  dynThresh = 512;  // MEGA 10-bit ADC
  rrCount = 0;
  lastPeakMs = 0;
}

void pulseUpdate() {
  unsigned long now = millis();
  if (now - lastPulseSampleMs < PULSE_SAMPLE_INTERVAL_MS) return;
  lastPulseSampleMs = now;

  int raw = analogRead(PIN_PULSE); // 0..1023

  // Glättung
  smoothSum -= smoothBuf[smoothIdx];
  smoothBuf[smoothIdx] = raw;
  smoothSum += raw;
  smoothIdx = (smoothIdx + 1) % SMOOTH_N;
  int smoothed = smoothSum / SMOOTH_N;

  // Dynamische Schwelle träge nachführen
  dynThresh = (dynThresh * 99 + smoothed) / 100;

  // Peak-Detektion
  if (!above && smoothed > dynThresh + 20) {      // Offset ggf. kalibrieren
    if (now - lastPeakMs > REFRACT_MS) {
      if (lastPeakMs > 0 && rrCount < RR_MAX) {
        rrBuf[rrCount++] = now - lastPeakMs;
      } else if (lastPeakMs > 0) {
        for (int i = 1; i < RR_MAX; i++) rrBuf[i - 1] = rrBuf[i];
        rrBuf[RR_MAX - 1] = now - lastPeakMs;
      }
      lastPeakMs = now;

      if (rrCount > 0) {
        unsigned long sum = 0;
        for (int i = 0; i < rrCount; i++) sum += rrBuf[i];
        float avgRR = sum / (float)rrCount;
        if (avgRR > 0) currentBPM = (int)round(60000.0f / avgRR);
      }
    }
    above = true;
  } else if (above && smoothed < dynThresh) {
    above = false;
  }
}

// ---------------- IMU ----------------
void imuInit() {
  Wire.begin();                 // MEGA: SDA=20, SCL=21
  // Optional schnellere I2C-Taktung, wenn stabil:
  // Wire.setClock(400000UL);

  imuReady = mpu.begin(0x69, &Wire);  // AD0=HIGH → 0x69
  if (imuReady) {
    Serial.println(F("[OK] MPU6050 auf Wire @ 0x69 erkannt."));
    mpu.setAccelerometerRange(MPU6050_RANGE_4_G);
    mpu.setGyroRange(MPU6050_RANGE_250_DEG);
    mpu.setFilterBandwidth(MPU6050_BAND_44_HZ);
  } else {
    Serial.println(F("[WARN] MPU6050 nicht gefunden – Verkabelung/Adresse/Spannung prüfen."));
  }
  lastIMU_ms = millis();
}

void imuUpdate() {
  if (!imuReady) return;

  unsigned long now = millis();
  float dt = (now - lastIMU_ms) / 1000.0f;
  if (dt <= 0) dt = 0.001f;
  lastIMU_ms = now;

  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  // Gyro-Raten in deg/s
  float gx = g.gyro.x * 180.0f / PI;
  float gy = g.gyro.y * 180.0f / PI;
  float gz = g.gyro.z * 180.0f / PI;

  // Yaw-Rate je nach Achse wählen
  float yaw_rate = (YAW_AXIS=='x' ? gx : (YAW_AXIS=='y' ? gy : gz));
  if (fabsf(yaw_rate) < YAW_RATE_DEADZONE) yaw_rate = 0.0f;

  // Integration + sanfter Zerfall
  yaw_deg += (-yaw_rate) * dt;
  yaw_deg *= expf(-dt / YAW_DECAY_TAU_S);

  // Pitch (einfache Schätzung aus Beschleunigung)
  const float g_const = 9.80665f;
  float axg = a.acceleration.x / g_const;
  float ayg = a.acceleration.y / g_const;
  float azg = a.acceleration.z / g_const;
  pitch_deg = atan2f(-axg, sqrtf(ayg * ayg + azg * azg)) * 180.0f / PI;

  // Kopfstatus
  String newStatus = headStatus;
  if (yaw_deg < GIER_L && headStatus != "links") {
    newStatus = "links"; headLastChangeMs = now;
  } else if (yaw_deg > GIER_R && headStatus != "rechts") {
    newStatus = "rechts"; headLastChangeMs = now;
  } else if (yaw_deg > GIER_L && yaw_deg < GIER_R && headStatus != "center") {
    newStatus = "center"; headLastChangeMs = now;
  }
  if (newStatus != headStatus) headStatus = newStatus;

  // Schweregrad (Pitch-Schwelle 20°)
  bool abgelenkt = ((headStatus == "links") || (headStatus == "rechts")) &&
                   ((now - headLastChangeMs) > (unsigned long)(ABLENKUNGS_ZEIT_S * 1000.0f));
  bool starkGeneigt = (pitch_deg > 20.0f) || (pitch_deg < -20.0f);

  if (starkGeneigt && abgelenkt)      sevGyro = SCHW_ERNST;
  else if (starkGeneigt || abgelenkt) sevGyro = SCHW_MITTEL;
  else                                 sevGyro = SCHW_NORMAL;
}

// ---------------- GSR ----------------
int readGSR() {
  return analogRead(PIN_GSR); // 0..1023 (MEGA 10-bit)
}

// Heuristische Schwellen (10-bit anpassen!)
Severity gsrToSeverity(int gsrRaw) {
  // Grobe Umrechnung von deinen 12-bit Schwellen:
  // 2200/4095 ≈ 0.537 → ~550 bei 10-bit; 1300/4095 ≈ 0.317 → ~325
  if (gsrRaw > 550) return SCHW_ERNST;
  if (gsrRaw > 325) return SCHW_MITTEL;
  return SCHW_NORMAL;
}

// ---------------- LED-Logik ----------------
void updateLEDs() {
  unsigned long now = millis();
  if (now - lastLEDUpdateMs < LED_AKT_INTERVAL_MS) return;
  lastLEDUpdateMs = now;

  // Puls relativ zu Baseline
  if (baselineBPM <= 0 || currentBPM == 0) {
    sevPulse = SCHW_NORMAL;
  } else if (currentBPM < baselineBPM - 5) {
    sevPulse = SCHW_ERNST;
  } else if (currentBPM < baselineBPM - 2) {
    sevPulse = SCHW_MITTEL;
  } else {
    sevPulse = SCHW_NORMAL;
  }

  int gsrRaw = readGSR();
  sevGSR = gsrToSeverity(gsrRaw);

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

// ---------------- Kalibrierung ----------------
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

// ---------------- Setup / Loop ----------------
void setup() {
  Serial.begin(115200);
  // Beim MEGA ist 'while(!Serial){}' nicht nötig; schadet aber nicht, wenn USB-Seriell genutzt wird.

  pinMode(PIN_LED_GRUEN, OUTPUT);
  pinMode(PIN_LED_GELB,  OUTPUT);
  pinMode(PIN_LED_ROT,   OUTPUT);
  setLED(true, true, true); delay(800); setLED(false, false, false);

  // MEGA: 10-bit ADC ist Standard → kein analogReadResolution()

  pulseInit();
  imuInit();
  calibratePulse(10000);

  Serial.println(F("millis,bpm,gyro_yaw_rate_dps,pitch_deg,yaw_deg,head_status,gsr_raw,gsr_level,led_state"));
}

void loop() {
  pulseUpdate();
  imuUpdate();
  updateLEDs();

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
