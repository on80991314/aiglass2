"""Find-and-grab state machine (5 states).

All coordinates normalized to [0, 1] in frame space (top-left origin).
"""

from __future__ import annotations

import time
import json
import threading
import concurrent.futures  # 👇 新增：用來建立持久化的執行緒池
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, List, Optional

from groq import Groq

from vision.detector import Detection
from vision.hands import HandResult

# --- 初始化 Groq ---
groq_client = Groq(api_key="GROQ_API_KEY")

# 👇 【終極防線：持久化執行緒池】
# 讓背景執行緒永遠活著，避免「執行緒死亡」時 Windows 誤刪 PyTorch 記憶體
llm_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

# 【防閃退機制 1】API 暖機：預先載入 SSL DLL，避免執行緒中途衝突閃退
try:
    print("[System] 正在進行 Groq API 暖機與網路模組載入...")
    groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": "hi"}],
        max_tokens=1
    )
    print("[System] Groq API 暖機完成！")
except Exception:
    pass

def analyze_voice_command(user_text: str):
    """結合 Groq 大腦與安全攔截的語意分析"""
    stop_keywords = ["結束", "停止", "不要", "關閉", "停", "取消"]
    confirm_keywords = ["拿到", "確認", "完成", "找到", "到", "到了", "好了"]

    if any(k in user_text for k in confirm_keywords):
        return "CONFIRM", None, None
    if any(k in user_text for k in stop_keywords):
        return "STOP", None, None

    prompt = f"""
    你是一個智慧導航眼鏡的指令分析大腦。請判斷使用者的語音意圖。
    如果使用者想要找物品 (例如：想要找杯子、幫我找手機)，請回傳 intent 為 "FIND_ITEM"，並提取他想找的「中文物品名」，同時將其翻譯成 YOLO 模型支援的「英文類別名」 (例如: cup, cell phone, bottle, laptop, bowl 等)。
    如果不是找物品，回傳 intent 為 "UNKNOWN"。

    請只回傳 JSON 格式，格式如下：
    {{"intent": "FIND_ITEM" 或 "UNKNOWN", "target_zh": "中文名", "target_en": "english_name"}}

    使用者說：「{user_text}」
    """
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile", # 換成更聰明的大模型，減少幻覺
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        data = json.loads(response.choices[0].message.content.strip())
        return data.get("intent", "UNKNOWN"), data.get("target_zh", ""), data.get("target_en", "")
    except Exception as e:
        print(f"[LLM Router Error] Groq API 發生錯誤: {e}")
        return "UNKNOWN", None, None


class FGState(Enum):
    WAITING_FOR_COMMAND = auto()
    SEARCHING_OBJECT = auto()
    GUIDING_HEAD = auto()
    GUIDING_HAND = auto()
    GRAB_SUCCESS = auto()


@dataclass
class FindGrabConfig:
    head_left_thresh: float = 0.4
    head_right_thresh: float = 0.6
    center_lo: float = 0.35
    center_hi: float = 0.65
    center_frames_required: int = 10
    hand_tolerance: float = 0.08       
    grab_radius: float = 0.10          
    iou_threshold: float = 0.30        
    grab_success_hold_s: float = 3.0
    head_prompt_interval_s: float = 1.2
    hand_prompt_interval_s: float = 0.6
    search_prompt_interval_s: float = 2.0
    object_memory_s: float = 2.0

@dataclass
class _Timers:
    last_head_prompt: float = 0.0
    last_hand_prompt: float = 0.0
    last_search_prompt: float = 0.0
    grab_success_start: float = 0.0


def _bbox_iou(a: tuple[float, float, float, float],
              b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


class FindGrabFSM:
    def __init__(self,
                 cfg: Optional[FindGrabConfig] = None,
                 emit: Optional[Callable[[str], None]] = None) -> None:
        self.cfg = cfg or FindGrabConfig()
        self._emit = emit or self._default_emit
        self.state: FGState = FGState.WAITING_FOR_COMMAND
        self.target_zh: Optional[str] = None
        self.target_en: Optional[str] = None
        self.center_streak: int = 0
        self.t = _Timers()
        self.last_obj: Optional[Detection] = None
        self.last_obj_time: float = 0.0
        self.last_stt_text: str = ""
        self.last_stt_time: float = 0.0

    def on_speech(self, text: str) -> None:
        """Feed STT output here."""
        self.last_stt_text = text
        self.last_stt_time = time.monotonic()
        self._emit(f"[STT] 收到語音辨識結果: {text!r}")

        # 👇 【關鍵修改】改用持久執行緒池，絕對不讓執行緒「出生又死亡」
        llm_executor.submit(self._process_llm_intent, text)

    def _process_llm_intent(self, text: str) -> None:
        """在背景執行緒中呼叫大腦，不會卡住系統，且執行緒做完會回去待命不會死掉"""
        intent, zh, en = analyze_voice_command(text)
        self._emit(f"[LLM ROUTER] 判斷結果: {intent} (zh={zh}, en={en})")

        if intent == "CONFIRM":
            if self.state != FGState.WAITING_FOR_COMMAND:
                self._emit(f"[FSM] 🎙️ 語音確認觸發：{text}")
                self.confirm_grab()
        elif intent == "STOP":
            if self.state != FGState.WAITING_FOR_COMMAND:
                self._emit(f"[FSM] 🎙️ 語音停止觸發：{text}")
                self._reset()
        elif intent == "FIND_ITEM" and zh and en:
            self.target_zh, self.target_en = zh, en
            self.center_streak = 0
            self.last_obj = None  
            self.t = _Timers()
            self._goto(FGState.SEARCHING_OBJECT)
            self._emit(f"[FSM] 🔄 收到新指令，切換目標為：{zh} ({en})")
        else:
            self._emit(f"[STT] 未辨識出尋物或確認指令: {text!r}")

    def confirm_grab(self) -> None:
        if self.state != FGState.WAITING_FOR_COMMAND:
            self._emit(f"[FSM] ✅ 已手動確認取得 {self.target_zh}！")
            self._reset()
    
    def set_target(self, zh: str, en: str) -> None:
        self.target_zh, self.target_en = zh, en
        self.center_streak = 0
        self.last_obj = None
        self.t = _Timers()
        self._goto(FGState.SEARCHING_OBJECT)
        self._emit(f"[FSM] 🎯 鍵盤直接指定目標：{zh} ({en})")

    def step(self, frame_size: tuple[int, int],
             dets: List[Detection],
             hand: Optional[HandResult]) -> None:
        w, h = frame_size
        if w <= 0 or h <= 0: return
        if self.state == FGState.WAITING_FOR_COMMAND: return
        if self.state == FGState.GRAB_SUCCESS:
            if time.monotonic() - self.t.grab_success_start >= self.cfg.grab_success_hold_s:
                self._reset()
            return

        current_obj = self._find_target(dets)
        now = time.monotonic()
        
        if current_obj is not None:
            self.last_obj = current_obj
            self.last_obj_time = now
            obj = current_obj
        else:
            if self.last_obj is not None and (now - self.last_obj_time) <= self.cfg.object_memory_s:
                obj = self.last_obj
            else:
                self.last_obj = None
                obj = None

        if self.state == FGState.SEARCHING_OBJECT:
            self._tick_search(obj)
        elif self.state == FGState.GUIDING_HEAD:
            self._tick_head(obj, w, h)
        elif self.state == FGState.GUIDING_HAND:
            self._tick_hand(obj, hand, w, h)

    def _tick_search(self, obj: Optional[Detection]) -> None:
        now = time.monotonic()
        if obj is None:
            if now - self.t.last_search_prompt >= self.cfg.search_prompt_interval_s:
                self.t.last_search_prompt = now
                self._emit(f"[FSM/SEARCH] 正在尋找 {self.target_zh}...")
            return
        self._emit(f"[FSM/SEARCH] 已發現 {self.target_zh}，進入頭部導引")
        self._goto(FGState.GUIDING_HEAD)

    def _tick_head(self, obj: Optional[Detection], w: int, h: int) -> None:
        if obj is None:
            self.center_streak = 0
            self._emit(f"[FSM/HEAD] 失去 {self.target_zh} 視線，重新搜尋")
            self._goto(FGState.SEARCHING_OBJECT)
            return

        x_norm = obj.cx / float(w)
        y_norm = obj.cy / float(h)

        if self.cfg.center_lo <= x_norm <= self.cfg.center_hi \
                and self.cfg.center_lo <= y_norm <= self.cfg.center_hi:
            self.center_streak += 1
        else:
            self.center_streak = 0

        now = time.monotonic()
        if now - self.t.last_head_prompt >= self.cfg.head_prompt_interval_s:
            self.t.last_head_prompt = now
            if x_norm < self.cfg.head_left_thresh:
                self._emit(f"[FSM/HEAD] {self.target_zh} 在你的左邊，請向左轉")
            elif x_norm > self.cfg.head_right_thresh:
                self._emit(f"[FSM/HEAD] {self.target_zh} 在你的右邊，請向右轉")
            else:
                self._emit(f"[FSM/HEAD] {self.target_zh} 就在你正前方")

        if self.center_streak >= self.cfg.center_frames_required:
            self._emit(f"[FSM/HEAD] 已穩定置中 {self.center_streak} 幀，進入手部導引")
            self._goto(FGState.GUIDING_HAND)

    def _tick_hand(self, obj: Optional[Detection],
                   hand: Optional[HandResult], w: int, h: int) -> None:
        if obj is None:
            self._emit(f"[FSM/HAND] 失去 {self.target_zh} 超過兩秒，退回頭部導引")
            self.center_streak = 0
            self._goto(FGState.GUIDING_HEAD)
            return
        if hand is None:
            now = time.monotonic()
            if now - self.t.last_hand_prompt >= self.cfg.hand_prompt_interval_s:
                self.t.last_hand_prompt = now
                self._emit("[FSM/HAND] 找不到你的手，請把手伸進畫面")
            return

        ox = obj.cx / float(w)
        oy = obj.cy / float(h)
        hx, hy = hand.tip_x, hand.tip_y
        dx = ox - hx  
        dy = oy - hy  

        dist = (dx * dx + dy * dy) ** 0.5
        obj_bbox = (obj.x1 / w, obj.y1 / h, obj.x2 / w, obj.y2 / h)
        iou = _bbox_iou(obj_bbox, hand.bbox_norm)

        if dist < self.cfg.grab_radius or iou >= self.cfg.iou_threshold:
            now = time.monotonic()
            if now - self.t.last_hand_prompt >= self.cfg.hand_prompt_interval_s:
                self.t.last_hand_prompt = now
                self._emit(f"[FSM/HAND] 有取得物品嗎，沒有的話物品在手的正前方")
            return

        now = time.monotonic()
        if now - self.t.last_hand_prompt < self.cfg.hand_prompt_interval_s:
            return
        self.t.last_hand_prompt = now

        tol = self.cfg.hand_tolerance
        msgs: list[str] = []
        if dx > tol: msgs.append("手向右移")
        elif dx < -tol: msgs.append("手向左移")
        if dy > tol: msgs.append("手向下移")
        elif dy < -tol: msgs.append("手向上移")
        if not msgs: msgs.append("位置已接近，慢慢靠過去")
        self._emit(f"[FSM/HAND] {'，'.join(msgs)} | d={dist:.3f} IoU={iou:.2f}")

    def _find_target(self, dets: List[Detection]) -> Optional[Detection]:
        if not self.target_en: return None
        cands = [d for d in dets if d.label.lower() == self.target_en.lower()]
        if not cands: return None
        return max(cands, key=lambda d: d.conf)

    def _goto(self, s: FGState) -> None:
        if s == self.state: return
        self.state = s
        if s == FGState.GRAB_SUCCESS:
            self.t.grab_success_start = time.monotonic()
            self._emit(f"[FSM] 進入 GRAB_SUCCESS — 已成功接觸 {self.target_zh}！")
        elif s == FGState.WAITING_FOR_COMMAND:
            self._emit("[FSM] 已重置，等待下一個指令")

    def _reset(self) -> None:
        self.state = FGState.WAITING_FOR_COMMAND
        self.target_zh = None
        self.target_en = None
        self.center_streak = 0
        self.last_obj = None
        self.t = _Timers()
        self._emit("[FSM] WAITING_FOR_COMMAND — 等待下一個指令 (按鍵盤 1~4)")

    @staticmethod
    def _default_emit(msg: str) -> None:
        print(msg, flush=True)