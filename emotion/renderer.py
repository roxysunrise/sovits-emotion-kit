"""renderer —— 把一句话编成「有情绪的 WAV」。

链路（严格按五段流水线的 ②③④，只是把生成层留给调用方注入）：

    text
      │ ① analyzer    → EmotionPoint（含 arc）+ 情绪惯性
      │ ② planner     → ProsodyPlan（逐小句曲线/气声/停顿）
      │ ③ synth_fn    → 每个小句调后端（edge / GPT-SoVITS / SAPI）拿到 PCM
      │ ④ dsp         → 变调曲线 + 颤音 + 颤栗 + 气声 + 句尾拖长 + 停顿 + 限幅
      ▼
    wav bytes + meta（含可选的客观指纹，用于质检回环）

synth_fn 的签名：(text, overrides) -> wav bytes
    overrides 里给后端能吃的参数：{"edge_rate": "+8%", "edge_pitch": "+12Hz",
    "gpt_sovits_speed": 1.08, "_emotion": {...}}
"""
from __future__ import annotations

from dataclasses import dataclass
import time

from . import analyzer as A
from . import dsp
from . import lexicon as L
from . import metrics
from . import persona as P
from . import planner as PL
from .schema import EmotionPoint

# 硬上限：QQ 语音单条 60s，留 2s 余量
MAX_SECONDS = 58.0


class TooLong(RuntimeError):
    """合成结果超过桥接能收的时长。"""


@dataclass
class RenderResult:
    wav: bytes
    meta: dict


def _prosody_override(point: EmotionPoint) -> dict:
    return point.__dict__.get("prosody_override") or {}


def apply_override(seg, point: EmotionPoint):
    """LLM 给的韵律偏移覆盖 planner 的默认值（只覆盖给了的字段）。"""
    ov = _prosody_override(point)
    if not ov:
        return seg
    if ov.get("rate_pct"):
        seg.rate_pct = max(-30.0, min(40.0, seg.rate_pct * 0.4 + ov["rate_pct"] * 0.6))
    if ov.get("semitones"):
        seg.base_semitones += ov["semitones"] * 0.5
    if ov.get("breath"):
        seg.breath = max(0.0, min(0.55, seg.breath * 0.5 + ov["breath"] * 0.5))
    if ov.get("pause_after"):
        seg.pause_after = ov["pause_after"]
    if ov.get("tail_lengthen", 1.0) > 1.0:
        seg.tail_lengthen = max(seg.tail_lengthen, ov["tail_lengthen"])
    if ov.get("vibrato_st"):
        seg.vibrato_st = max(seg.vibrato_st, ov["vibrato_st"])
    return seg


def render(
    text: str,
    synth_fn,
    *,
    persona: dict | None = None,
    lang: str = "zh",
    base_f0: float = 220.0,
    prior: EmotionPoint | None = None,
    hint_label: str | None = None,
    llm_conf: dict | None = None,
    emotion: EmotionPoint | None = None,
    use_emoji: bool = False,
    backend_dsp: bool = True,
    tail_trim_db: float = -45.0,
    with_metrics: bool = False,
    max_seconds: float = MAX_SECONDS,
    log=print,
) -> RenderResult:
    """返回 (wav, meta)。synth_fn 由调用方提供（worker 里就是后端派发）。"""
    p = persona or P.load()
    t0 = time.time()
    clean, _hints = A.extract_hints(text or "")
    # 小洛不用 emoji / 表情包：送后端之前先把表情符号和 [表情包:xx] 这类标记剔掉，
    # 否则 TTS 会把「😭」念成怪东西，或者把整个占位标记念出来
    clean, cleaned = L.clean_for_speech(clean, keep_emoji=use_emoji)
    clauses = A.split_clauses(clean) or [clean or "嗯"]
    if emotion is None:
        # 注意：这里要喂**原始文本**，不是已清洗的 clean。
        # 因为「【难过】」「（小声）」这类内联表演指令是给分析器看的，
        # 一旦先剥掉，情绪就判不出来了 —— 之前就是这么丢的。
        emotion = A.analyze(text or "", prior=prior, hint_label=hint_label,
                            use_llm=llm_conf, use_emoji=use_emoji)
    plan = PL.build_plan(clauses, emotion, p, lang=lang, base_f0=base_f0)

    audio_parts: list[list[float]] = []
    rate_out = 24000
    seg_meta: list[dict] = []
    for i, seg in enumerate(plan.segments):
        pt = emotion.arc[i] if i < len(emotion.arc) else emotion
        apply_override(seg, pt)
        overrides = {
            "emotion": emotion.to_dict(),
            "segment": seg.to_dict(),
            "edge_rate": _pct(seg.rate_pct),
            "_emotion_label": seg.label,
            "gpt_sovits_speed": round(1.0 + seg.rate_pct / 100.0, 3),
        }
        raw = synth_fn(seg.text, overrides)
        x, rate_out = dsp.parse_wav(raw)
        x = dsp.trim_silence(x, rate_out)
        # 模型的混响尾巴（末字拖很远）听感就是"回音"：剪掉 -45dB 以下的弱拖尾。
        # backend_dsp 走不走都做 —— 这不是效果，是去尾巴。阈值 ≤ -90 视为关闭。
        if tail_trim_db > -90:
            x = dsp.trim_tail(x, rate_out, thresh_db=tail_trim_db)

        if backend_dsp:
            # ① 音高曲线 + 颤音 + 颤栗
            x = dsp.pitch_contour(
                x, rate_out,
                contour=seg.contour,
                base_semitones=seg.base_semitones,
                vibrato_hz=seg.vibrato_hz,
                vibrato_st=seg.vibrato_st,
                tremor=seg.tremor,
                seed=7 + i,
            )
            # ② 句尾拖长（软、舍不得、委屈）
            if seg.tail_lengthen > 1.001:
                x = dsp.stretch_tail(x, rate_out, seg.tail_lengthen, last_ms=200.0)
            # ③ 气声（先做，别把后面要留白的停顿糊上噪声）
            x = dsp.add_breath(x, rate_out, seg.breath, seed=11 + i)
            # ④ 句内换气：柔软/低唤醒的情绪在最安静处塞一个微停顿（犹豫、哽咽）
            if seg.breath > 0.22 or seg.label in ("难过", "疲惫", "哭腔", "委屈", "温柔"):
                hesitate = 0.08 + 0.20 * max(0.0, 1.0 - seg.intensity * 0.5) * (0.5 + seg.breath)
                x = dsp.insert_at_quiet(x, rate_out, min(0.30, hesitate))
        # backend_dsp=False（GPT-SoVITS 自带情感）：情绪已经由参考音频决定，
        # 这里再做移调/气声会把模型刚学到的韵律抹平，还会让音色发飘。
        # ⑤ 能量：先归一到 -20dBFS，再按情绪给增益（顺序反了会把情绪差异抹平）
        x = dsp.normalize_rms(x, -20.0)
        # 段间增益锁在 ±1.6dB：小句各自增益差太大（实测难过段 −2.5dB）
        # 在长句里会听成"忽大忽小"，也就是用户说的重音。
        gain = max(-1.6, min(1.6, float(seg.gain_db)))
        x = dsp.apply_gain(x, gain)

        audio_parts.append(x)
        audio_parts.append(dsp.silence(rate_out, seg.pause_after if i < len(plan.segments) - 1 else 0.02))
        seg_meta.append({
            "text": seg.text, "label": seg.label, "intensity": seg.intensity,
            "rate_pct": seg.rate_pct, "semitones": seg.base_semitones,
            "breath": seg.breath, "tremor": seg.tremor,
            "pause": seg.pause_after, "tail": seg.tail_lengthen,
            "sec": round(len(x) / float(rate_out), 3),
        })

    mono = dsp.crossfade_concat(audio_parts, rate_out, fade_ms=10.0)
    # 整段响度：以 -19dBFS 为基准，再按整段情绪能量偏一点，保留「兴奋更响 / 低落更轻」
    avg_gain = sum(s.gain_db for s in plan.segments) / max(1, len(plan.segments))
    mono = dsp.normalize_rms(mono, -19.0 + max(-3.5, min(3.5, avg_gain * 0.60)))

    # 超长保护：先压缩停顿，再报错让调用方降级
    dur = len(mono) / float(rate_out)
    if dur > max_seconds:
        log(f"    [!] 情感渲染 {dur:.1f}s 超过 {max_seconds:.0f}s，压缩停顿并截断")
        for s in plan.segments:
            s.pause_after = min(s.pause_after, 0.10)
        # 简单处理：截断到上限（语音消息里宁可短，不要发不出去）
        mono = mono[: int(max_seconds * rate_out)]
        dur = len(mono) / float(rate_out)

    wav = dsp.build_wav(mono, rate_out)
    meta = {
        "emotion": emotion.to_dict(),
        "plan": plan.to_dict(),
        "segments": seg_meta,
        "lang": lang,
        "duration": round(dur, 3),
        "ms": int((time.time() - t0) * 1000),
        "cleaned": cleaned,
        "backend_dsp": backend_dsp,
        "notes": plan.notes,
    }
    if with_metrics:
        meta["metrics"] = metrics.fingerprint(mono, rate_out)
    return RenderResult(wav=wav, meta=meta)


def _pct(v: float) -> str:
    return f"{'+' if v >= 0 else ''}{int(round(v))}%"