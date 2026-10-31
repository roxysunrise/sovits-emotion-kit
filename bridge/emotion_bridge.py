"""emotion_bridge —— 把情感库挂到**已有的** TTS 路由/网关上（可选件）。

如果你的链路已经是「某个服务收文本 → 调 GPT-SoVITS → 返回 wav」，
不用换成 server/emotion_server.py，只在这里接入两行即可：

    import emotion_bridge
    emo = emotion_bridge.emotion_for(segment_text, {"enabled": True,
                                                    "voice_dir": r"D:/path/to/kit"})
    body.update({k: v for k, v in emo.items() if not k.startswith("_")})
    # 语音的快慢：body["speed_factor"] = base_speed * emo["speed_factor"]

它只做一件事：**按情绪换参考音频和采样参数**，不碰你的模型、不碰你的音频。
情感库载入失败时返回空 dict，你的链路照旧用固定 ref 出声。

真实接入示例见 bridge/SERVICE_PATCH.md。
"""
from __future__ import annotations

import os
import sys
import threading

_LOCK = threading.Lock()
_STATE: dict = {"tried": False, "ok": False, "err": "", "an": None, "refs": None}


def _load(voice_dir: str) -> bool:
    with _LOCK:
        if _STATE["tried"]:
            return _STATE["ok"]
        _STATE["tried"] = True
        try:
            if voice_dir not in sys.path:
                sys.path.insert(0, voice_dir)
            from emotion import analyzer as _an  # noqa: WPS433
            from emotion import refs as _refs    # noqa: WPS433
            _STATE["an"] = _an
            _STATE["refs"] = _refs
            _STATE["ok"] = True
        except Exception as error:  # noqa: BLE001
            _STATE["err"] = f"{type(error).__name__}: {error}"
            _STATE["ok"] = False
        return _STATE["ok"]


def status() -> dict:
    return {"ok": _STATE["ok"], "err": _STATE["err"]}


def emotion_for(text: str, cfg: dict) -> dict:
    """返回要覆盖进请求体的字段；空 dict = 不动（用你原来的固定 ref）。

    cfg:
        enabled         开关
        voice_dir       本 kit 的根目录（含 emotion/ 包）
        refs_dir        情绪 ref 目录（相对 voice_dir 或绝对路径）
        clean_refs      true = 不改 ref，只调采样参数（音质最稳）
        min_confidence  低于这个置信度就按「平静」处理
        speed           额外语速系数
    """
    if not cfg or not cfg.get("enabled"):
        return {}
    voice_dir = cfg.get("voice_dir") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if not _load(voice_dir):
        return {}
    an, refs = _STATE["an"], _STATE["refs"]
    try:
        point = an.analyze(text or "", hint_label=cfg.get("hint_label") or None)
    except Exception:  # noqa: BLE001
        return {}
    label = getattr(point, "label", "") or "平静"
    intensity = float(getattr(point, "intensity", 0.3) or 0.3)
    if float(getattr(point, "confidence", 1.0) or 1.0) < float(cfg.get("min_confidence", 0.0)):
        label = "平静"

    refs_dir = cfg.get("refs_dir") or "refs/emotional"
    if not os.path.isabs(refs_dir):
        refs_dir = os.path.join(voice_dir, refs_dir)
    pick = refs.pick(label, intensity, speed=float(cfg.get("speed", 1.0) or 1.0),
                     out_dir=refs_dir, clean_only=bool(cfg.get("clean_refs", False)))
    if not pick:
        return {}
    return {
        "ref_audio_path": str(pick.get("gpt_sovits_ref_audio", "")).replace("\\", "/"),
        "prompt_text": pick.get("gpt_sovits_prompt_text", ""),
        "prompt_lang": pick.get("gpt_sovits_prompt_lang", "all_ja"),
        "temperature": float(pick.get("gpt_sovits_temperature", 1.0)),
        "top_k": int(pick.get("gpt_sovits_top_k", 15)),
        "top_p": float(pick.get("gpt_sovits_top_p", 1.0)),
        "repetition_penalty": float(pick.get("gpt_sovits_rep_pen", 1.35)),
        "fragment_interval": float(pick.get("gpt_sovits_fragment_interval", 0.3)),
        "speed_factor": float(pick.get("gpt_sovits_speed", 1.0)),
        "_emotion": label,
        "_intensity": round(intensity, 3),
    }