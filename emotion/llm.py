"""可选 LLM 精标 —— 让上游大模型直接给结构化情感（OpenAI 兼容 /chat/completions）。

纪律（导出对话里的「情感标签一定要结构化」）：
    只要 JSON，不要「有点忧伤又有点期待」这种散文；
    字段范围硬校验，失败就退回规则分析器，绝不因为 LLM 挂了导致不出声。

配置示例（worker config.json）：
    "emotion": {
      "llm": {
        "enabled": true,
        "url": "http://127.0.0.1:11434/v1/chat/completions",
        "model": "qwen2.5:7b",
        "api_key": "",
        "timeout": 6
      }
    }
"""
from __future__ import annotations

import json
import re
import urllib.request

from .schema import EmotionPoint
from .lexicon import LABELS, LABEL_ALIASES

SYSTEM = "你是语音表演导演。只输出 JSON，不要解释，不要 markdown。"

PROMPT = """给这句话配情感参数。标注角色「小洛」：黏人、甜、爱撒娇，中文少女声。

文本：{text}

输出 JSON，字段严格如下：
{{"label": "下列之一 {labels}", "valence": -1~1, "arousal": 0~1,
 "dominance": 0~1, "intensity": 0~1,
 "prosody": {{"rate_pct": -25~35, "semitones": -3~3,
             "breath": 0~0.5, "pause_after": 0.05~0.6,
             "tail_lengthen": 1.0~1.5, "vibrato_st": 0~0.4}}}}

规则：
- valence 负面必须是负数；anger 的 dominance 要高（>0.7），委屈要低（<0.2）。
- 只要数值，不要字符串单位。"""


def _post(url: str, payload: dict, api_key: str, timeout: float) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {api_key}"} if api_key else {}),
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def _extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        return json.loads(m.group(0))
    raise ValueError("LLM 没给 JSON")


def _num(d: dict, key: str, lo: float, hi: float, default: float) -> float:
    try:
        v = float(d.get(key, default))
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, v))


def annotate(text: str, rule_point: EmotionPoint, conf: dict) -> EmotionPoint | None:
    """拿 LLM 的标注，失败返回 None（调用方退回规则结果）。"""
    if not conf.get("enabled"):
        return None
    url = conf.get("url") or ""
    if not url:
        return None
    model = conf.get("model") or "gpt-4o-mini"
    timeout = float(conf.get("timeout", 6))
    prompt = PROMPT.format(text=text[:500], labels="、".join(LABELS))
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 220,
        "stream": False,
    }
    if conf.get("response_format_json", True):
        payload["response_format"] = {"type": "json_object"}
    got = _post(url, payload, conf.get("api_key", ""), timeout)
    content = ""
    try:
        content = got["choices"][0]["message"]["content"]
    except Exception:
        content = got.get("response") or got.get("content") or ""
    if not content:
        return None
    data = _extract_json(content)

    label = str(data.get("label", "")).strip()
    label = LABEL_ALIASES.get(label.lower(), label)
    if label not in LABELS:
        label = rule_point.label

    p = data.get("prosody") or {}
    pt = EmotionPoint(
        valence=_num(data, "valence", -1.0, 1.0, rule_point.valence),
        arousal=_num(data, "arousal", 0.0, 1.0, rule_point.arousal),
        dominance=_num(data, "dominance", 0.0, 1.0, rule_point.dominance),
        intensity=_num(data, "intensity", 0.0, 1.0, rule_point.intensity),
        label=label,
        confidence=0.85,
        source="llm",
    ).normalized()
    # 把 LLM 给的韵律偏移塞进 arc 的第一项（renderer 会读 prosody_override）
    pt.__dict__["prosody_override"] = {
        "rate_pct": _num(p, "rate_pct", -25, 35, 0.0),
        "semitones": _num(p, "semitones", -3, 3, 0.0),
        "breath": _num(p, "breath", 0.0, 0.5, 0.0),
        "pause_after": _num(p, "pause_after", 0.05, 0.6, 0.0),
        "tail_lengthen": _num(p, "tail_lengthen", 1.0, 1.5, 1.0),
        "vibrato_st": _num(p, "vibrato_st", 0.0, 0.4, 0.0),
    }
    return pt