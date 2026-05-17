"""Find-and-grab pipeline entry point.

Two video sources:
   --source webcam         (default, cv2.VideoCapture(0))
   --source ws             (receive JPEG frames from ESP32-S3 over WebSocket)
"""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# 👇 【終極防線 1】在載入任何 AI 套件前，強制先喚醒 Windows 的網路與 SSL 模組
import ssl
import httpx
from groq import Groq
_ = ssl.create_default_context()  # 提早觸發底層 C 函式庫載入，避免中途打架

import faulthandler
faulthandler.enable()

# 👇 網路鋪好路後，接著才載入 PyTorch 並限制
import torch
torch.set_num_threads(1)
torch.set_grad_enabled(False)  # 👇 新增這行：徹底關閉訓練引擎，防止記憶體碎片化

# 👇 最後載入 OpenCV
import cv2
cv2.setNumThreads(1)

import argparse
import asyncio
import logging
import sys
import threading
import time
import queue
from pathlib import Path
from typing import List, Optional

import numpy as np

# make sibling packages importable when run as `python edge/find_grab_main.py`
sys.path.insert(0, str(Path(__file__).resolve().parent))

from audio.mic import MicListener  # noqa: E402
from state_machine.find_grab import FGState, FindGrabConfig, FindGrabFSM  # noqa: E402
from utils.logging_setup import setup as setup_logging  # noqa: E402
from vision.detector import Detection, YoloDetector  # noqa: E402
from vision.hands import HandResult, HandsDetector  # noqa: E402
from PIL import Image, ImageDraw, ImageFont


log = logging.getLogger("find_grab")


# ----- shared between threads -----
_latest_frame: Optional[np.ndarray] = None
_latest_frame_lock = threading.Lock()
_stop_flag = threading.Event()
_ws_audio_queue = queue.Queue()  # 👇 新增：用來存放 ESP32 傳來的音訊

def set_latest(frame: np.ndarray) -> None:
    global _latest_frame
    with _latest_frame_lock:
        _latest_frame = frame


def get_latest() -> Optional[np.ndarray]:
    with _latest_frame_lock:
        return None if _latest_frame is None else _latest_frame.copy()

class Esp32AudioListener:
    def __init__(self, on_text: Callable[[str], None], stt) -> None:
        self.on_text = on_text
        self.stt = stt
        self.thread = threading.Thread(target=self._run, daemon=True, name="esp32_audio")
        self.audio_buffer = bytearray()
        self.pre_buffer = bytearray() # 👇 新增：用來存放講話前 0.5 秒的聲音
        
        self.speaking = False
        self.last_voice_time = time.monotonic()
        self.start_talk_time = time.monotonic()
        self.energy_thresh = 300 

    def start(self) -> None: self.thread.start()
    def stop(self) -> None: pass

    def _run(self) -> None:
        log.info("ESP32 Audio listener started (Smart VAD Mode V2)")
        max_record_time = 6.0  
        
        while not _stop_flag.is_set():
            try:
                data = _ws_audio_queue.get(timeout=0.1)
                chunk_np = np.frombuffer(data, dtype=np.int16)
                if len(chunk_np) == 0: continue
                
                chunk_float = chunk_np.astype(np.float32)
                chunk_ac = chunk_float - np.mean(chunk_float)
                energy = np.mean(np.abs(chunk_ac)) 
                now = time.monotonic()

                if energy > self.energy_thresh:
                    if not self.speaking:
                        self.speaking = True
                        self.audio_buffer.clear()
                        # 👇 關鍵修改：把講話前 0.5 秒的聲音接回來，防止「拿」被吃掉
                        self.audio_buffer.extend(self.pre_buffer)
                        print(f"🎙️ 開始錄音... (起伏音量: {int(energy)})")
                        self.start_talk_time = now
                    self.last_voice_time = now
                    self.audio_buffer.extend(data)
                
                elif self.speaking:
                    self.audio_buffer.extend(data)
                    if now - self.last_voice_time > 0.8 or now - self.start_talk_time > max_record_time:
                        print(f"✅ 錄音結束，開始辨識... (長度: {now - self.start_talk_time:.1f}秒)")
                        self._process_buffer()
                else:
                    # 👇 關鍵修改：沒講話時，隨時保留最後 0.5 秒的聲音 (16000 取樣率 * 2 bytes * 0.5 秒 = 16000)
                    self.pre_buffer.extend(data)
                    if len(self.pre_buffer) > 16000:
                        self.pre_buffer = self.pre_buffer[-16000:]

            except queue.Empty:
                if self.speaking and (time.monotonic() - self.last_voice_time > 0.8):
                    print("✅ 錄音結束 (無新資料)，開始辨識...")
                    self._process_buffer()
                continue
            except Exception as e:
                log.error("STT Audio processing error: %s", e)
                self.audio_buffer.clear()
                self.speaking = False

    def _process_buffer(self):
        if len(self.audio_buffer) > 16000:
            audio_np = np.frombuffer(bytes(self.audio_buffer), dtype=np.int16)
            audio_f32 = audio_np.astype(np.float32) * 3.0
            audio_np = np.clip(audio_f32, -32768, 32767).astype(np.int16)
            
            text = self.stt.transcribe_pcm(audio_np, sample_rate=16000)
            if text and text.strip():
                self.on_text(text.strip())
        
        self.audio_buffer.clear()
        self.pre_buffer.clear() # 清空預錄區
        self.speaking = False

# ------------ video sources ------------
def webcam_producer(index: int = 0) -> None:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        log.error("webcam open failed (index=%d)", index)
        _stop_flag.set()
        return
    try:
        while not _stop_flag.is_set():
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            set_latest(frame)
    finally:
        cap.release()


async def _ws_handler(ws) -> None:
    log.info("ESP32 connected: %s", ws.remote_address)
    try:
        async for msg in ws:
            if not isinstance(msg, (bytes, bytearray)):
                continue
            data = bytes(msg)
            if data[:2] == b"\xff\xd8":     # phase-1 raw JPEG
                jpeg = data
            elif data and data[0] == 0x01:  # phase-3 tagged JPEG
                jpeg = data[1:]
            elif data and data[0] == 0x02:  # 👇 新增：phase-4 音訊資料 (以 0x02 開頭)
                _ws_audio_queue.put(data[1:])
                continue
            else:
                continue
            
            arr = np.frombuffer(jpeg, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is not None:
                set_latest(frame)
    finally:
        log.info("ESP32 disconnected")


def ws_producer(host: str, port: int) -> None:
    import websockets

    async def _serve():
        async def _h(ws):
            await _ws_handler(ws)
        server = await websockets.serve(_h, host=host, port=port,
                                        max_size=8 * 1024 * 1024,
                                        ping_interval=20, ping_timeout=20)
        log.info("WebSocket source listening on ws://%s:%d", host, port)
        try:
            while not _stop_flag.is_set():
                await asyncio.sleep(0.2)
        finally:
            server.close()
            await server.wait_closed()

    asyncio.run(_serve())

# 👇 新增：建立一個全域的字體快取字典，避免每幀重複讀取硬碟
_FONT_CACHE = {}

def put_chinese_text(img, text, position, text_color=(0, 255, 255), font_size=24):
    if not text:
        return img
        
    # 轉換為 PIL Image
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # 👇 修改：只在第一次需要時載入字體，之後直接從快取拿
    global _FONT_CACHE
    if font_size not in _FONT_CACHE:
        try:
            # 載入 Windows 內建的微軟正黑體
            _FONT_CACHE[font_size] = ImageFont.truetype("msjh.ttc", font_size)
        except IOError:
            _FONT_CACHE[font_size] = ImageFont.load_default()
            
    font = _FONT_CACHE[font_size]
        
    draw.text(position, text, font=font, fill=text_color)
    # 轉回 OpenCV 格式
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ------------ main loop ------------
def draw_overlay(frame: np.ndarray,
                 fsm: FindGrabFSM,
                 dets: List[Detection],
                 target_det: Optional[Detection],
                 hand: Optional[HandResult]) -> np.ndarray:
    h, w = frame.shape[:2]
    for d in dets:
        color = (0, 255, 255) if d is target_det else (100, 200, 100)
        cv2.rectangle(frame, (d.x1, d.y1), (d.x2, d.y2), color, 2)
        cv2.putText(frame, f"{d.label} {d.conf:.2f}", (d.x1, max(18, d.y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    if hand is not None:
        hx1 = int(hand.x1 * w); hy1 = int(hand.y1 * h)
        hx2 = int(hand.x2 * w); hy2 = int(hand.y2 * h)
        cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), (255, 0, 255), 2)
        tip = (int(hand.tip_x * w), int(hand.tip_y * h))
        cv2.circle(frame, tip, 6, (255, 0, 255), -1)
        cv2.putText(frame, "hand", (hx1, max(18, hy1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)

    cv2.rectangle(frame, (int(0.35 * w), int(0.35 * h)),
                  (int(0.65 * w), int(0.65 * h)), (80, 80, 80), 1)

    banner = f"STATE: {fsm.state.name}  target: {fsm.target_zh or '-'}" \
             f"  streak: {fsm.center_streak}"
    cv2.rectangle(frame, (0, 0), (w, 24), (0, 0, 0), -1)
    cv2.putText(frame, banner, (8, 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    # 👇 新增：在畫面底部顯示 STT 語音字幕 (停留 4 秒)
    now = time.monotonic()
    if now - fsm.last_stt_time < 4.0 and fsm.last_stt_text:
        text_to_show = f"語音輸入: {fsm.last_stt_text}"
        
        # 畫一個半透明黑色背景條，讓字體更清楚
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 50), (w, h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # 寫上中文字幕
        frame = put_chinese_text(frame, text_to_show, (20, h - 40), text_color=(0, 255, 255), font_size=24)
    return frame


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", choices=["webcam", "ws"], default="webcam")
    p.add_argument("--webcam-index", type=int, default=0)
    p.add_argument("--ws-host", default="0.0.0.0")
    p.add_argument("--ws-port", type=int, default=8081)
    p.add_argument("--yolo-weights", default="yolov8n.pt")
    p.add_argument("--yolo-device", default="cpu")
    p.add_argument("--yolo-conf", type=float, default=0.35)
    p.add_argument("--mic", action="store_true", help="enable microphone STT")
    p.add_argument("--stt-model", default="tiny")
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    setup_logging(args.log_level)
    # 👇 加入這行：建立一把全域的 AI 鎖
    detector = YoloDetector(args.yolo_weights, args.yolo_device, args.yolo_conf)
    hands = HandsDetector(max_num_hands=1)
    fsm = FindGrabFSM(FindGrabConfig())

    if args.source == "webcam":
        t_src = threading.Thread(target=webcam_producer, args=(args.webcam_index,),
                                 daemon=True, name="webcam")
    else:
        t_src = threading.Thread(target=ws_producer, args=(args.ws_host, args.ws_port),
                                 daemon=True, name="ws")
    t_src.start()

    mic = None
    if args.mic:
        from audio.stt import WhisperSTT
        stt = WhisperSTT(model_size=args.stt_model, language="zh")
        
        if args.source == "ws":
            mic = Esp32AudioListener(on_text=fsm.on_speech, stt=stt)
            log.info("ESP32 mic enabled — try saying 「幫我找手機」 or 「我拿到了」")
        else:
            mic = MicListener(on_text=fsm.on_speech, stt=stt)
            log.info("PC mic enabled — try saying 「幫我找手機」 or 「我拿到了」")
            
        mic.start()
    else:
        log.info("mic disabled — press 't' in the window to simulate 「想要找杯子」")

    window = "Find & Grab (q to quit)"
    fsm._emit("[FSM] WAITING_FOR_COMMAND — 請說「想要找 XX」")

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 800, 600)

    # 👇 新增：用來記錄上一幀的結果與交替開關
    last_dets = []
    last_hand = None
    frame_toggle = True

    try:
        while not _stop_flag.is_set():
            frame = get_latest()
            if frame is None:
                time.sleep(0.01)
                cv2.waitKey(1)
                continue

            sys.stdout.flush()
            frame = cv2.flip(frame, 1)
            frame = np.ascontiguousarray(frame)
            
            run_hands = fsm.state == FGState.GUIDING_HAND

            # 👇 【終極殺招：交替幀運算】徹底錯開 YOLO 和 MediaPipe
            if not run_hands:
                # 1. 還沒進入手部導引時：全力跑 YOLO
                dets = detector.infer(frame.copy())
                last_dets = dets
                hand = None
                last_hand = None
            else:
                # 2. 進入手部導引時：單雙數幀交替跑，絕對不讓 CPU 打架
                if frame_toggle:
                    # 這幀只跑 YOLO，手部沿用上次紀錄
                    dets = detector.infer(frame.copy())
                    last_dets = dets
                    hand = last_hand
                else:
                    # 這幀只跑 MediaPipe，物品沿用上次紀錄
                    dets = last_dets
                    mp_frame = np.array(frame, copy=True, order='C')
                    hand = hands.detect(mp_frame)
                    last_hand = hand
                
                # 切換開關 (True 變 False, False 變 True)
                frame_toggle = not frame_toggle

            sys.stdout.flush()
            fsm.step((frame.shape[1], frame.shape[0]), dets, hand)

            #print("5. 準備繪製並顯示畫面...")
            sys.stdout.flush()
            target_det = fsm._find_target(dets) if fsm.target_en else None
            overlay = draw_overlay(frame, fsm, dets, target_det, hand)
            cv2.imshow(window, overlay)
            # 在畫面上印出當前時間，如果時間還在跳，代表主程式沒死，是 ESP32 斷線了
            cv2.putText(overlay, f"Time: {time.time():.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # 👆 --- 替換到這裡為止 ---

            k = cv2.waitKey(1) & 0xFF
            if k in (ord('q'), 27):
                break
            elif k == ord('r'):
                fsm._reset()
            elif k == ord('c'):
                fsm.confirm_grab()
            elif k == ord('t'):
                fsm.set_target("杯子", "cup")
                
            # 👇 改成直接呼叫 set_target，並同時給定中文和 YOLO 英文標籤
            elif k == ord('1'):
                fsm.set_target("水壺", "bottle")
            elif k == ord('2'):
                fsm.set_target("手機", "cell phone")
            elif k == ord('3'):
                fsm.set_target("筆電", "laptop")
            elif k == ord('4'):
                fsm.set_target("碗", "bowl")
    finally:
        _stop_flag.set()
        if mic:
            mic.stop()
        hands.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
