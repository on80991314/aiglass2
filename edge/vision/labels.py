"""Chinese -> COCO-English label mapping used when parsing voice commands."""

from __future__ import annotations

import re

ZH_TO_COCO = {
    "人": "person", "行人": "person",
    "腳踏車": "bicycle", "自行車": "bicycle",
    "機車": "motorcycle", "摩托車": "motorcycle",
    "汽車": "car", "車": "car", "車子": "car",
    "公車": "bus", "巴士": "bus",
    "貨車": "truck", "卡車": "truck",
    "紅綠燈": "traffic light",
    "停車號誌": "stop sign",
    "背包": "backpack",
    "手提包": "handbag", "包包": "handbag",
    "行李箱": "suitcase",
    "雨傘": "umbrella", "傘": "umbrella",
    "領帶": "tie",
    "飛盤": "frisbee",
    "滑雪板": "skis",
    "滑板": "skateboard",
    "衝浪板": "surfboard",
    "球拍": "tennis racket", "網球拍": "tennis racket",
    "水瓶": "bottle", "瓶子": "bottle",
    "紅酒杯": "wine glass", "酒杯": "wine glass",
    "杯子": "cup", "馬克杯": "cup",
    "叉子": "fork",
    "刀子": "knife", "刀": "knife",
    "湯匙": "spoon", "匙": "spoon",
    "碗": "bowl",
    "香蕉": "banana",
    "蘋果": "apple",
    "三明治": "sandwich",
    "橘子": "orange", "柳橙": "orange",
    "青花菜": "broccoli", "花椰菜": "broccoli",
    "紅蘿蔔": "carrot", "胡蘿蔔": "carrot",
    "熱狗": "hot dog",
    "披薩": "pizza",
    "甜甜圈": "donut",
    "蛋糕": "cake",
    "椅子": "chair",
    "沙發": "couch",
    "盆栽": "potted plant", "植物": "potted plant",
    "床": "bed",
    "餐桌": "dining table", "桌子": "dining table", "桌": "dining table",
    "馬桶": "toilet",
    "電視": "tv", "電視機": "tv",
    "筆電": "laptop", "筆記型電腦": "laptop", "電腦": "laptop",
    "滑鼠": "mouse",
    "遙控器": "remote",
    "鍵盤": "keyboard",
    "手機": "cell phone", "電話": "cell phone", "行動電話": "cell phone",
    "微波爐": "microwave",
    "烤箱": "oven",
    "烤麵包機": "toaster",
    "水槽": "sink",
    "冰箱": "refrigerator",
    "書": "book", "書本": "book",
    "時鐘": "clock", "鐘": "clock",
    "花瓶": "vase",
    "剪刀": "scissors",
    "泰迪熊": "teddy bear", "熊娃娃": "teddy bear",
    "吹風機": "hair drier", "吹風機": "hair drier",
    "牙刷": "toothbrush",
}


# Common noise words the user may say around the object noun.
_STRIP_PREFIXES = ["我的", "那個", "那支", "那隻", "一個", "一支", "一隻", "這個"]

_PATTERNS = [
    re.compile(r"想要?找[一個個支隻]?(.+?)(?:[，。！？]|$)"),
    re.compile(r"幫我找[一個個支隻]?(.+?)(?:[，。！？]|$)"),
    re.compile(r"找(.+?)(?:[，。！？]|$)"),
]


def parse_find_command(text: str) -> tuple[str, str] | None:
    """Return (target_zh, target_en) if the utterance matches a find-command."""
    if not text:
        return None
    for pat in _PATTERNS:
        m = pat.search(text)
        if not m:
            continue
        raw = m.group(1).strip()
        for pref in _STRIP_PREFIXES:
            if raw.startswith(pref):
                raw = raw[len(pref):].strip()
        if not raw:
            continue
        en = ZH_TO_COCO.get(raw)
        if en:
            return raw, en
        # fallback: maybe user said the English label directly
        low = raw.lower()
        if low in set(ZH_TO_COCO.values()):
            return raw, low
    return None
