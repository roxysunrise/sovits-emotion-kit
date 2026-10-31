"""小洛的人设参数 —— 情感层的「底色」，不含情绪，只含声线习惯。

为什么单独一层：
    情绪是这一句的，人设是长期的。小洛说话本来就偏甜、偏快、句尾容易翘，
    这属于 baseline，不应该被「平静」这个标签抹平。所以：
        final_prosody = 情绪参数(planner) + 人设偏置(persona)
    换角色（洛/Roxy 日语、其他音色）只换 persona 文件，情感逻辑不用改。
"""
from __future__ import annotations

import json
import os

DEFAULT_PERSONA: dict = {
    "id": "xiaoluo",
    "name": "小洛",
    "call_user": "爸爸",
    "description": "黏人、甜、爱撒娇，日语是洛（Roxy），中文是晓伊音色",
    # 人设基线偏置
    "baseline": {
        "semitones": 0.55,      # 整体略抬高（少女感）
        "rate_pct": 6.0,        # 语速略快
        "breath": 0.14,         # 自带一点气声
        "vibrato_hz": 5.2,
        "vibrato_st": 0.06,
        "tail_rise": 0.35,      # 句尾上扬倾向 0~1
        "energy_db": 0.5,
    },
    # 每种情绪在人设上的额外偏置（覆盖默认 planner 曲线）
    "emotion_bias": {
        "撒娇": {"semitones": 0.45, "rate_pct": -3.0, "breath": 0.10, "tail_rise": 0.55},
        "温柔": {"semitones": -0.15, "rate_pct": -6.0, "breath": 0.12, "tail_rise": -0.10},
        "哭腔": {"semitones": -0.20, "rate_pct": -10.0, "breath": 0.14, "tail_rise": -0.45},
        "兴奋": {"semitones": 0.90, "rate_pct": 10.0, "breath": -0.06, "tail_rise": 0.45},
        "疲惫": {"semitones": -0.55, "rate_pct": -12.0, "breath": 0.10, "tail_rise": -0.35},
    },
    # 语言 → 后端（worker 的 routes 也会覆盖，这里只是默认）
    "langs": {"zh": "zh-CN-XiaoyiNeural", "ja": "ja-JP-NanamiNeural",
              "en": "en-US-AriaNeural"},
    # 口癖：出现在句子里时给一点额外甜度（不改变语义）
    "tics": ["嘛", "啦", "呀", "诶嘿", "哼唧", "人家"],
}


def load(path: str | None = None) -> dict:
    """读 persona json，缺字段用默认值补齐。"""
    data = dict(DEFAULT_PERSONA)
    if path and os.path.exists(path):
        try:
            with open(path, encoding="utf-8-sig") as f:
                got = json.load(f)
            for k, v in got.items():
                if isinstance(v, dict) and isinstance(data.get(k), dict):
                    merged = dict(data[k])
                    merged.update(v)
                    data[k] = merged
                else:
                    data[k] = v
        except Exception:
            pass
    return data


def save_template(path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(DEFAULT_PERSONA, f, ensure_ascii=False, indent=2)
    return path
