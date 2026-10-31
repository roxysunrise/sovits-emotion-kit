"""persona —— 声线底色（可选，默认中性）。

情感层只做「情绪 → 韵律 → ref」的加工，不带任何角色设定。
如果你想固定某种声线习惯（整体略高/略快/气声多一点），用 JSON 覆盖这里的默认值：

    final_prosody = 情绪参数(planner) + 声线偏置(persona)

默认全 0 = 不加任何偏置，情绪曲线原样生效。
"""
from __future__ import annotations

import json
import os

DEFAULT_PERSONA: dict = {
    "id": "default",
    "name": "",
    "description": "中性声线，无额外偏置",
    "baseline": {
        "semitones": 0.0,
        "rate_pct": 0.0,
        "breath": 0.0,
        "vibrato_hz": 5.0,
        "vibrato_st": 0.05,
        "tail_rise": 0.0,
        "energy_db": 0.0,
    },
    "emotion_bias": {},
    "tics": [],
}


def load(path: str | None = None) -> dict:
    """读 persona json，缺字段用默认值补齐；不给路径就是纯中性。"""
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