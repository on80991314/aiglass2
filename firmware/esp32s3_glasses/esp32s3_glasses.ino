/*
 * AI Smart Glasses - Phase 3+ integrated firmware.
 *
 * On one WebSocket connection:
 *   TX tag 0x01 + JPEG           (camera)
 *   TX tag 0x02 + <LE u32 SR> + int16 PCM   (mic)
 *   TX text JSON {"type":"imu",...}
 *   RX tag 0x03 + <LE u32 SR> + int16 PCM   (speaker)
 *   RX text JSON {"type":"cmd",...}
 *
 * Lib deps (install via Arduino Library Manager):
 *   - ArduinoWebsockets by Gil Maimon
 *   - ArduinoJson       by Benoit Blanchon
 *   - Adafruit MPU6050 + Adafruit Unified Sensor
 */

#include <Arduino.h>
#include <WiFi.h>
#include <Wire.h>
#include <ArduinoWebsockets.h>
#include <ArduinoJson.h>
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#define sensor_t camera_sensor_t
#include "esp_camera.h"
#undef sensor_t
#include "ESP_I2S.h"

#include "wifi_config.h"
#include "camera_pins.h"
#include "hw_config.h"

#define TAG_JPEG          0x01
#define TAG_AUDIO_UP      0x02
#define TAG_AUDIO_DOWN    0x03

using namespace websockets;

I2SClass i2sIn;
I2SClass i2sOut;
static WebsocketsClient ws;
static Adafruit_MPU6050 mpu;
static bool imuReady = false;
static uint32_t lastReconnectMs = 0;

// ---- camera ----
static bool initCamera() {
  camera_config_t c = {};
  c.ledc_channel = LEDC_CHANNEL_0;
  c.ledc_timer   = LEDC_TIMER_0;
  c.pin_d0 = Y2_GPIO_NUM; c.pin_d1 = Y3_GPIO_NUM; c.pin_d2 = Y4_GPIO_NUM; c.pin_d3 = Y5_GPIO_NUM;
  c.pin_d4 = Y6_GPIO_NUM; c.pin_d5 = Y7_GPIO_NUM; c.pin_d6 = Y8_GPIO_NUM; c.pin_d7 = Y9_GPIO_NUM;
  c.pin_xclk = XCLK_GPIO_NUM; c.pin_pclk = PCLK_GPIO_NUM;
  c.pin_vsync = VSYNC_GPIO_NUM; c.pin_href = HREF_GPIO_NUM;
  c.pin_sccb_sda = SIOD_GPIO_NUM; c.pin_sccb_scl = SIOC_GPIO_NUM;
  c.pin_pwdn = PWDN_GPIO_NUM; c.pin_reset = RESET_GPIO_NUM;
  c.xclk_freq_hz = 20000000;
  c.pixel_format = PIXFORMAT_JPEG;
  if (psramFound()) {
    c.frame_size = FRAMESIZE_VGA; c.jpeg_quality = 12; c.fb_count = 2;
    c.fb_location = CAMERA_FB_IN_PSRAM; c.grab_mode = CAMERA_GRAB_LATEST;
  } else {
    c.frame_size = FRAMESIZE_QVGA; c.jpeg_quality = 15; c.fb_count = 1;
    c.fb_location = CAMERA_FB_IN_DRAM; c.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
  }
  esp_err_t err = esp_camera_init(&c);
  if (err != ESP_OK) return false;
  camera_sensor_t *s = esp_camera_sensor_get();
  if (s) {
    s->set_vflip(s, CAM_VFLIP);
    s->set_hmirror(s, CAM_HMIRROR);
  }
  return true;
}

// ---- I2S mic (PDM 模式 - XIAO ESP32S3 內建) ----
static bool initMic() {
  i2sIn.setPinsPdmRx(42, 41); // XIAO 內建麥克風腳位: CLK=42, DATA=41
  // 設定為 PDM 接收、16kHz、16-bit 單聲道
  if (!i2sIn.begin(I2S_MODE_PDM_RX, MIC_SAMPLE_RATE, I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO)) {
    return false;
  }
  return true;
}

// ---- I2S speaker (如果有的話) ----
static bool initSpeaker() {
  i2sOut.setPins(SPK_BCLK_PIN, SPK_LRC_PIN, SPK_DIN_PIN);
  if (!i2sOut.begin(I2S_MODE_STD, SPK_SAMPLE_RATE, I2S_DATA_BIT_WIDTH_16BIT, I2S_SLOT_MODE_MONO)) {
    return false;
  }
  return true;
}

static void sendJpeg() {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) return;
  if (fb->format == PIXFORMAT_JPEG && fb->len > 0) {
    uint8_t *pkt = (uint8_t*)malloc(fb->len + 1);
    if (pkt) {
      pkt[0] = TAG_JPEG;
      memcpy(pkt + 1, fb->buf, fb->len);
      ws.sendBinary((const char*)pkt, fb->len + 1);
      free(pkt);
    }
  }
  esp_camera_fb_return(fb);
}

static void sendMicChunk() {
  // 每次最多傳送 40ms 的資料 (16000Hz * 0.04s = 640 samples)
  const size_t max_samples = MIC_SAMPLE_RATE / 1000 * MIC_FRAME_MS;
  static int16_t buf[max_samples];
  size_t samples_read = 0;

  // 👇 關鍵修改：使用非阻塞 (Non-blocking) 的逐次讀取
  while (samples_read < max_samples) {
    int v = i2sIn.read();
    if (v == -1) break; // -1 代表目前 I2S 緩衝區空了，立刻跳出，絕不卡死 loop！
    buf[samples_read++] = (int16_t)v;
  }

  // 如果有讀到聲音，就打包傳出去
  if (samples_read > 0) {
    size_t bytes_len = samples_read * 2;
    size_t plen = 1 + bytes_len;
    uint8_t *pkt = (uint8_t*)malloc(plen);
    if (pkt) {
      pkt[0] = TAG_AUDIO_UP;
      memcpy(pkt + 1, buf, bytes_len);
      ws.sendBinary((const char*)pkt, plen);
      free(pkt);
    }
  }
}

static void sendImu() {
  if (!imuReady) return;
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);
  StaticJsonDocument<192> doc;
  doc["type"] = "imu";
  doc["t"] = millis() / 1000.0;
  doc["ax"] = a.acceleration.x;
  doc["ay"] = a.acceleration.y;
  doc["az"] = a.acceleration.z;
  doc["gx"] = g.gyro.x;
  doc["gy"] = g.gyro.y;
  doc["gz"] = g.gyro.z;
  char buf[192];
  size_t n = serializeJson(doc, buf, sizeof(buf));
  ws.send(buf, n);
}

static void playAudioDown(const uint8_t *data, size_t len) {
  if (len < 5) return;
  // 略過前面 5 bytes 的協定標頭，將剩下的音訊寫入喇叭
  i2sOut.write((uint8_t*)(data + 5), len - 5);
}

static void onWsMessage(WebsocketsMessage msg) {
  if (msg.isBinary()) {
    const auto &raw = msg.rawData();
    if (raw.size() > 0 && (uint8_t)raw[0] == TAG_AUDIO_DOWN) {
      playAudioDown((const uint8_t*)raw.data(), raw.size());
    }
  } else if (msg.isText()) {
    Serial.printf("[WS] cmd: %s\n", msg.c_str());
  }
}

static void onWsEvent(WebsocketsEvent event, String data) {
  switch (event) {
    case WebsocketsEvent::ConnectionOpened:   Serial.println("[WS] connected"); break;
    case WebsocketsEvent::ConnectionClosed:   Serial.println("[WS] disconnected"); break;
    case WebsocketsEvent::GotPing:            break;
    case WebsocketsEvent::GotPong:            break;
  }
}

static void connectWifi() {
  WiFi.mode(WIFI_STA);
  WiFi.setSleep(false);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("[WiFi] connect");
  uint32_t t0 = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - t0 < 20000) { delay(250); Serial.print('.'); }
  Serial.println();
  if (WiFi.status() == WL_CONNECTED) { Serial.print("[WiFi] IP "); Serial.println(WiFi.localIP()); }
}

static bool wsConnect() {
  char url[96];
  snprintf(url, sizeof(url), "ws://%s:%u%s", WS_HOST, WS_PORT, WS_PATH);
  return ws.connect(url);
}

void setup() {
  Serial.begin(115200);
  delay(200);
  Serial.println("\n=== AI Smart Glasses (integrated) ===");

  if (!initCamera())  { Serial.println("[CAM] init fail"); }
  if (!initMic())     { Serial.println("[MIC] init fail"); }
  //if (!initSpeaker()) { Serial.println("[SPK] init fail"); }

  Wire.begin(IMU_SDA_PIN, IMU_SCL_PIN);
  if (mpu.begin(IMU_I2C_ADDR)) {
    mpu.setAccelerometerRange(MPU6050_RANGE_4_G);
    mpu.setGyroRange(MPU6050_RANGE_500_DEG);
    mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
    imuReady = true;
    Serial.println("[IMU] ready");
  } else {
    Serial.println("[IMU] not found (skip)");
  }

  connectWifi();

  ws.onMessage(onWsMessage);
  ws.onEvent(onWsEvent);
  wsConnect();
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) { delay(20); return; }

  if (ws.available()) {
    ws.poll();
  } else if (millis() - lastReconnectMs > 2000) {
    lastReconnectMs = millis();
    wsConnect();
    return;
  }

  static uint32_t tVid = 0, tImu = 0, tMic = 0;
  uint32_t now = millis();

  if (now - tVid >= TARGET_FRAME_INTERVAL_MS) { tVid = now; sendJpeg(); }
  if (now - tMic >= MIC_FRAME_MS)             { tMic = now; sendMicChunk(); }
  if (now - tImu >= (1000 / IMU_SAMPLE_HZ))   { tImu = now; sendImu(); }
}
