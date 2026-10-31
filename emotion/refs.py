"""refs —— GPT-SoVITS 的「情感参考库」。

为什么要有这个模块
------------------
GPT-SoVITS 的情绪不是靠参数开关，而是靠 **参考音频（in-context learning）**：
给它一段什么情绪的 ref + 对应的 prompt_text，它就用那个音色、那种韵律去念你要的文本。
所以「让 SoVITS 自己带情感」的正确做法是：

    emotion.analyze（判情绪） → 选对应的 ref + 采样参数 → 模型直接念出情绪

**不是**先让模型念成中性、再对波形做变调/气声后处理（那是另一条 edge 路线的做法，
放在 SoVITS 上会把它自带的韵律抹平，还会让音色发飘）。

ref 从哪来
----------
1. 首选：真人/洛的原声素材，每个情绪录 3~8 秒，写进 index.json（最像本人）。
2. 兜底：从一段中性 ref 用 DSP 派生（移调 + 变速 + 气声 + 增益）。
   派生版的音色会轻微偏移（±2~3 半音），但情绪方向和韵律是真实的 ——
   实测 6 个派生 ref 喂给 9880，输出的 F0/语速/气声差异明显（见 README 表格）。

用法
----
    from emotion import refs
    refs.derive_all("D:/.../roxy-reference.wav", "voice/refs/emotional")   # 生成/刷新
    ov = refs.pick("难过", intensity=0.6)                                   # 拿到参数
    # ov 直接 merge 进 worker 的 overrides（键名已是 gpt_sovits_*）

命令行
------
    python -m emotion.refs derive --base BASE.wav --out refs/emotional
    python -m emotion.refs list  --dir refs/emotional
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

from . import dsp, metrics

# ── 情绪原型 ────────────────────────────────────────────────
# 17 个细标签收敛到 10 个 ref 原型：ref 越多越难维护，而且相邻情绪的 ref 差异
# 本来就小于合成噪声（实测 生气 vs 急切 在同一 ref 下几乎听不出差别）。
PROTOS = ("平静", "开心", "撒娇", "温柔", "难过", "生气", "疲惫", "冷淡", "惊讶", "害怕")

ALIAS = {
    "兴奋": "开心", "调皮": "撒娇", "害羞": "撒娇",
    "感动": "温柔", "委屈": "难过", "哭腔": "难过",
    "急切": "生气",
}

# ── 派生参数（从一段中性 ref 造出各情绪的 ref）────────────────
# semitones: 相对基准的音高偏移（用相位声码器做，音质干净）
# gain_db:   能量微调
# ⚠️ 对 ref **不做变速、不加气声**：实测这两种处理会改变谱平坦度（0.44→0.71），
#    而 GPT-SoVITS 会把 ref 的噪声/抖动当音色学走 → 输出就是"电音"。
#    快慢交给采样参数里的 speed_factor（模型内部做，不产生伪影），气声交给
#    temperature / repetition_penalty。breath / speed 字段保留只为兼容旧索引。
DERIVE: dict[str, dict[str, float]] = {
    "平静": dict(semitones=0.0, speed=1.00, breath=0.00, gain_db=0.0),
    "开心": dict(semitones=2.2, speed=1.00, breath=0.00, gain_db=1.2),
    "撒娇": dict(semitones=3.0, speed=1.00, breath=0.00, gain_db=0.4),
    "温柔": dict(semitones=-0.8, speed=1.00, breath=0.00, gain_db=-0.8),
    "难过": dict(semitones=-2.0, speed=1.00, breath=0.00, gain_db=-1.6),
    "生气": dict(semitones=1.0, speed=1.00, breath=0.00, gain_db=2.2),
    "疲惫": dict(semitones=-1.5, speed=1.00, breath=0.00, gain_db=-2.0),
    "冷淡": dict(semitones=-1.0, speed=1.00, breath=0.00, gain_db=-1.2),
    "惊讶": dict(semitones=3.5, speed=1.00, breath=0.00, gain_db=0.8),
    "害怕": dict(semitones=1.8, speed=1.00, breath=0.00, gain_db=-0.4),
}

# ── 采样参数（模型侧的表现力，强度越大越偏）──────────────────
# temperature 高 → 音高/时长抖动更多（听着更"活"）；sad 要低一点，否则会飘。
SAMPLING: dict[str, dict[str, float]] = {
    "平静": dict(temperature=0.90, top_k=15, top_p=1.00, rep=1.35, fragment_interval=0.30, speed=1.00),
    "开心": dict(temperature=1.05, top_k=18, top_p=1.00, rep=1.30, fragment_interval=0.22, speed=1.06),
    "撒娇": dict(temperature=1.02, top_k=16, top_p=1.00, rep=1.32, fragment_interval=0.28, speed=1.02),
    "温柔": dict(temperature=0.94, top_k=14, top_p=1.00, rep=1.38, fragment_interval=0.34, speed=0.95),
    "难过": dict(temperature=0.92, top_k=14, top_p=0.98, rep=1.40, fragment_interval=0.46, speed=0.90),
    "生气": dict(temperature=1.00, top_k=18, top_p=1.00, rep=1.28, fragment_interval=0.24, speed=1.10),
    "疲惫": dict(temperature=0.90, top_k=12, top_p=0.98, rep=1.42, fragment_interval=0.52, speed=0.86),
    "冷淡": dict(temperature=0.88, top_k=12, top_p=0.98, rep=1.45, fragment_interval=0.38, speed=0.98),
    "惊讶": dict(temperature=1.08, top_k=20, top_p=1.00, rep=1.25, fragment_interval=0.20, speed=1.12),
    "害怕": dict(temperature=1.00, top_k=16, top_p=1.00, rep=1.32, fragment_interval=0.44, speed=1.06),
}

DEFAULT_DIR = "refs/emotional"
INDEX_NAME = "index.json"


def proto(label: str) -> str:
    """细标签 → ref 原型。未知标签落到 平静。"""
    if label in PROTOS:
        return label
    return ALIAS.get(label, "平静")


def _abs(p: str) -> str:
    return os.path.abspath(p)


def time_scale(x: list[float], rate: int, factor: float) -> list[float]:
    """OLA 变速（不改音高）。ref 只是给模型的提示，40ms 帧够用。"""
    if abs(factor - 1.0) < 0.02:
        return x
    frame = max(64, int(rate * 0.040))
    hop_a = max(32, int(rate * 0.020))
    hop_s = max(1, int(hop_a * factor))
    win = dsp._hann(frame)
    out = [0.0] * (int(len(x) * factor) + frame)
    pos = 0
    i = 0
    while i + frame < len(x):
        for j in range(frame):
            out[pos + j] += x[i + j] * win[j]
        i += hop_a
        pos += hop_s
    return out[: int(len(x) * factor)]


def derive_one(src_wav: str, out_path: str, *, semitones: float = 0.0, speed: float = 1.0,
               breath: float = 0.0, gain_db: float = 0.0, mode: str = "wsola") -> str:
    """从基准 ref 派生一个情绪 ref。

    mode="wsola"（默认）：重采样 + WSOLA 拉回时长 —— 变调但时长不变，
        没有相位声码器的"嗡嗡"、也没有颗粒法的帧率调制。ref 用它最干净。
    mode="pv"      ：相位声码器（音质可接受，但幅度调制偏大，约原始的两倍）
    mode="grain"   ：旧颗粒法（会带电子颤音，留作对比）
    speed / breath 默认不作用到 ref（见 DERIVE 上方注释）。
    """
    with open(src_wav, "rb") as f:
        x, rate = dsp.parse_wav(f.read())
    x = dsp.trim_silence(x, rate)
    if abs(semitones) > 0.01:
        if mode == "grain":
            x = dsp.pitch_contour(x, rate, contour=[(0.0, 0.0), (1.0, 0.0)],
                                  base_semitones=semitones, vibrato_hz=0.0,
                                  vibrato_st=0.0, tremor=0.0, seed=3)
        elif mode == "pv":
            x = dsp.pitch_shift_pv(x, rate, semitones)
        else:
            x = dsp.pitch_shift_wsola(x, rate, semitones)
    if breath > 0:
        # 只有在调用方明确要求时才加（默认 0）：加噪会把 ref 变"沙"，模型会学走
        x = dsp.add_breath(x, rate, breath, seed=5)
    if abs(speed - 1.0) > 0.02:
        x = time_scale(x, rate, 1.0 / speed)   # 默认不触发；仅自定义时用
    x = dsp.normalize_rms(x, -20.0)
    x = dsp.apply_gain(x, gain_db)
    os.makedirs(os.path.dirname(_abs(out_path)) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(dsp.build_wav(x, rate))
    return out_path


def validate(path: str) -> tuple[bool, dict]:
    """ref 能不能用：3~12 秒、单声道优先、采样率 ≥16k、别整段静音。"""
    info: dict[str, Any] = {"path": path}
    if not os.path.exists(path):
        return False, {**info, "error": "文件不存在"}
    try:
        with open(path, "rb") as f:
            x, rate = dsp.parse_wav(f.read())
    except Exception as e:
        return False, {**info, "error": f"不是合法 WAV：{e}"}
    sec = len(x) / float(rate)
    info.update(sec=round(sec, 2), rate=rate, rms_db=round(dsp.rms_db(x), 1))
    if sec < 3.0:
        return False, {**info, "error": f"太短（{sec:.1f}s < 3s）"}
    if sec > 12.0:
        return False, {**info, "error": f"太长（{sec:.1f}s > 12s，模型会慢且更飘）"}
    if rate < 16000:
        return False, {**info, "error": f"采样率偏低（{rate}）"}
    if info["rms_db"] < -45:
        return False, {**info, "error": "基本是静音"}
    return True, info


def derive_all(base_wav: str, out_dir: str = DEFAULT_DIR, *,
               prompt_text: str = "", prompt_lang: str = "all_ja",
               force: bool = False) -> dict:
    """生成整套情绪 ref + index.json。已有就跳过（force=True 才重做）。"""
    out_dir = _abs(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    index_path = os.path.join(out_dir, INDEX_NAME)
    old = {}
    if os.path.exists(index_path) and not force:
        try:
            old = json.load(open(index_path, encoding="utf-8-sig"))
        except Exception:
            old = {}
    refs: dict[str, dict] = {}
    for name in PROTOS:
        path = os.path.join(out_dir, f"{name}.wav")
        if force or not os.path.exists(path):
            derive_one(base_wav, path, **DERIVE[name])
        ok, info = validate(path)
        with open(path, "rb") as f:
            x, rate = dsp.parse_wav(f.read())
        fp = metrics.fingerprint(x, rate)
        refs[name] = {
            "file": path.replace("\\", "/"),
            "prompt_text": prompt_text or (old.get("refs", {}).get(name, {}).get("prompt_text") or ""),
            "prompt_lang": prompt_lang,
            "ok": ok,
            "problem": "" if ok else info.get("error", ""),
            "sec": info.get("sec"),
            "f0_med": round(fp["f0_med"], 1),
            "rms_db": round(fp["rms_db"], 1),
            "hf_ratio": round(fp["hf_ratio"], 3),
            "derived": {"base": _abs(base_wav).replace("\\", "/"), **DERIVE[name]},
        }
    index = {
        "version": 1,
        "built_at": int(time.time()),
        "base": _abs(base_wav).replace("\\", "/"),
        "prompt_text": prompt_text,
        "prompt_lang": prompt_lang,
        "refs": refs,
    }
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    return index


def load_index(out_dir: str = DEFAULT_DIR) -> dict:
    p = os.path.join(_abs(out_dir), INDEX_NAME)
    if not os.path.exists(p):
        return {}
    try:
        return json.load(open(p, encoding="utf-8-sig"))
    except Exception:
        return {}


def pick(label: str, intensity: float = 0.3, *, speed: float = 1.0,
         out_dir: str = DEFAULT_DIR, lang: str | None = None,
         clean_only: bool = False) -> dict:
    """情绪 → GPT-SoVITS 请求参数。键名已经是 worker 的 gpt_sovits_* 口径。

    lang 给了但和 index 里的 prompt_lang 不一致时，prompt_lang 仍然沿用 index
    （ref 是什么语言就得说什么语言；文本语言由调用方的 text_lang 决定）。

    clean_only=True：**不改 ref**（永远用基准 ref），情绪只靠采样参数表达 ——
    音质绝对干净，代价是情绪幅度小。config 里 `emotion.gsv.clean_refs=true` 启用。
    """
    idx = load_index(out_dir)
    name = proto(label)
    if clean_only:
        name = "平静"
    entry = (idx.get("refs") or {}).get(name) or {}
    path = entry.get("file") or os.path.join(_abs(out_dir), f"{name}.wav").replace("\\", "/")
    if not os.path.exists(path):
        return {}
    s = SAMPLING[proto(label)]
    it = max(0.0, min(1.0, float(intensity)))
    # 强度往「更偏」推：温度最多 +0.06，语速最多 ±6%，停顿最多 +0.10s
    ov = {
        "gpt_sovits_ref_audio": path,
        "gpt_sovits_prompt_text": entry.get("prompt_text", ""),
        "gpt_sovits_prompt_lang": entry.get("prompt_lang", "all_ja"),
        "gpt_sovits_speed": round(s["speed"] * speed * (1.0 + 0.06 * it * (1 if s["speed"] >= 1 else -1)), 3),
        "gpt_sovits_temperature": round(min(1.15, s["temperature"] + 0.06 * it), 3),
        "gpt_sovits_top_k": int(s["top_k"]),
        "gpt_sovits_top_p": float(s["top_p"]),
        "gpt_sovits_rep_pen": float(max(1.15, s["rep"] - 0.05 * it)),
        "gpt_sovits_fragment_interval": round(s["fragment_interval"] + 0.10 * it, 3),
        "_gsv_proto": name,
    }
    return ov


def stats(out_dir: str = DEFAULT_DIR) -> list[dict]:
    """自检用：每个 ref 的情绪指纹，能看出派生是不是真做了（F0/气声要单调）。"""
    rows = []
    for name in PROTOS:
        p = os.path.join(_abs(out_dir), f"{name}.wav")
        if not os.path.exists(p):
            rows.append({"label": name, "ok": False, "problem": "缺文件"})
            continue
        ok, info = validate(p)
        try:
            with open(p, "rb") as f:
                x, rate = dsp.parse_wav(f.read())
            fp = metrics.fingerprint(x, rate)
        except Exception:
            fp = {}
        rows.append({"label": name, "ok": ok, "sec": info.get("sec"),
                     "f0_med": round(fp.get("f0_med", 0), 1),
                     "rms_db": round(fp.get("rms_db", 0), 1),
                     "hf_ratio": round(fp.get("hf_ratio", 0), 3),
                     "problem": info.get("error", "")})
    return rows


def _cli() -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="emotion.refs", description="情绪参考音频库（GPT-SoVITS）")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("derive", help="从基准 ref 派生整套情绪 ref")
    d.add_argument("--base", required=True)
    d.add_argument("--out", default=DEFAULT_DIR)
    d.add_argument("--prompt-text", default="")
    d.add_argument("--prompt-lang", default="all_ja")
    d.add_argument("--force", action="store_true")
    l = sub.add_parser("list", help="列出 ref 与情绪指纹")
    l.add_argument("--dir", dest="out", default=DEFAULT_DIR)
    v = sub.add_parser("validate", help="检查某个 ref 能不能用")
    v.add_argument("path")
    a = ap.parse_args()
    if a.cmd == "derive":
        idx = derive_all(a.base, a.out, prompt_text=a.prompt_text,
                         prompt_lang=a.prompt_lang, force=a.force)
        print(f"已生成 {len(idx['refs'])} 个 ref → {os.path.abspath(a.out)}")
        for row in stats(a.out):
            print(f"  {row['label']:4s} {str(row['sec']):>5s}s  F0={row['f0_med']:6.1f}Hz  "
                  f"rms={row['rms_db']:6.1f}dB  hf={row['hf_ratio']:.3f}  "
                  f"{'OK' if row['ok'] else row['problem']}")
        return 0
    if a.cmd == "list":
        for row in stats(a.out):
            print(f"  {row['label']:4s} {str(row['sec']):>5s}s  F0={row['f0_med']:6.1f}Hz  "
                  f"rms={row['rms_db']:6.1f}dB  hf={row['hf_ratio']:.3f}  "
                  f"{'OK' if row['ok'] else row['problem']}")
        return 0
    ok, info = validate(a.path)
    print(json.dumps({**info, "ok": ok}, ensure_ascii=False, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
