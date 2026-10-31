"""情感数据结构 —— 小洛语音情感层的公共契约。

设计原则（来自导出对话里的五段流水线）：
    接入层只搬文本 → 理解层产出「结构化情感」→ 生成层吃参数 → 质检层量指标。

所以这里只定三样东西，全部可 JSON 序列化，方便 qqagent / worker / 外部 LLM 之间传：
    1. EmotionPoint   某一句话的情感点（valence-arousal-dominance + 强度 + 标签）
    2. ProsodyPlan    这句话最终要变成的韵律参数（后端参数 + DSP 参数 + 逐段曲线）
    3. SegmentPlan    逐句/逐小句的落地方案（含停顿、拖长、气声、颤音）

坐标系用 VAD 三维 + 强度，不用离散标签硬编码：
    valence   效价   -1（极负面） … +1（极正面）
    arousal   唤醒    0（平静/低落）… 1（激动）
    dominance 掌控    0（弱势/委屈）… 1（强势/生气）
    intensity 强度    0（淡）… 1（浓）
这样「委屈」= V略负 A中 D极低，「生气」= V负 A高 D高，机器能连续映射到参数。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Iterable


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


# ───────────────────────── 情感点 ─────────────────────────

@dataclass
class EmotionPoint:
    valence: float = 0.0
    arousal: float = 0.25
    dominance: float = 0.5
    intensity: float = 0.25
    label: str = "平静"
    confidence: float = 0.5
    # 逐小句的情感弧：[(小句文本, EmotionPoint)]，用来做「一句话内的情绪走向」
    arc: list["EmotionPoint"] = field(default_factory=list)
    source: str = "rule"       # rule / llm / hint / bridge / state

    def normalized(self) -> "EmotionPoint":
        self.valence = _clamp(self.valence, -1.0, 1.0)
        self.arousal = _clamp(self.arousal, 0.0, 1.0)
        self.dominance = _clamp(self.dominance, 0.0, 1.0)
        self.intensity = _clamp(self.intensity, 0.0, 1.0)
        self.confidence = _clamp(self.confidence, 0.0, 1.0)
        return self

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["arc"] = [a.to_dict() for a in self.arc]
        return d

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "EmotionPoint":
        arc = [EmotionPoint.from_dict(a) for a in (d.get("arc") or [])]
        p = EmotionPoint(
            valence=float(d.get("valence", 0.0)),
            arousal=float(d.get("arousal", 0.25)),
            dominance=float(d.get("dominance", 0.5)),
            intensity=float(d.get("intensity", 0.25)),
            label=str(d.get("label", "平静")),
            confidence=float(d.get("confidence", 0.5)),
            source=str(d.get("source", "bridge")),
        )
        p.arc = arc
        return p.normalized()

    # ── 向量运算，给状态机用 ──
    def vec(self) -> tuple[float, float, float, float]:
        return (self.valence, self.arousal, self.dominance, self.intensity)

    def lerp(self, other: "EmotionPoint", alpha: float) -> "EmotionPoint":
        """self*(1-alpha) + other*alpha，alpha 越大越贴近 other。"""
        a = _clamp(alpha, 0.0, 1.0)
        return EmotionPoint(
            valence=self.valence * (1 - a) + other.valence * a,
            arousal=self.arousal * (1 - a) + other.arousal * a,
            dominance=self.dominance * (1 - a) + other.dominance * a,
            intensity=self.intensity * (1 - a) + other.intensity * a,
            label=other.label if a >= 0.5 else self.label,
            confidence=max(self.confidence, other.confidence),
            source=f"state:{self.label}->{other.label}",
        ).normalized()

    def decay(self, toward: "EmotionPoint", rate: float) -> "EmotionPoint":
        """情绪回落：向基线靠 rate。"""
        return self.lerp(toward, _clamp(rate, 0.0, 1.0))


# ───────────────────────── 韵律方案 ─────────────────────────

@dataclass
class SegmentPlan:
    """一个小句怎么念。"""
    text: str
    # 后端参数（TTS 自己能做的）
    rate_pct: float = 0.0          # 语速偏移，%，喂 edge 的 +8% / GSV 的 speed_factor
    base_semitones: float = 0.0    # 整句基础移调，半音（DSP 精调，后端只做大粗调）
    gain_db: float = 0.0
    # 情感曲线
    contour: list[tuple[float, float]] = field(default_factory=lambda: [(0.0, 0.0), (1.0, 0.0)])
    vibrato_hz: float = 0.0
    vibrato_st: float = 0.0
    tremor: float = 0.0            # 颤栗（哭腔/害怕），0~0.35
    breath: float = 0.0            # 气声比例 0~0.6
    # 句读
    pause_after: float = 0.12
    tail_lengthen: float = 1.0     # 句尾拖长倍数，1.0 = 不拖
    label: str = "平静"
    intensity: float = 0.25

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ProsodyPlan:
    emotion: EmotionPoint
    segments: list[SegmentPlan] = field(default_factory=list)
    # 给后端用的汇总（edge 的 rate/pitch 字符串在这里生成）
    backend: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "emotion": self.emotion.to_dict(),
            "segments": [s.to_dict() for s in self.segments],
            "backend": self.backend,
            "notes": self.notes,
        }


def edge_pct(v: float) -> str:
    """edge-tts 的 rate 参数：'+12%' / '-8%'。"""
    return f"{'+' if v >= 0 else ''}{int(round(v))}%"


def edge_hz(st: float, base_f0: float = 220.0) -> str:
    """半音 → edge-tts 的 Hz 偏移（以女声基频 220Hz 折算）。"""
    hz = base_f0 * (2.0 ** (st / 12.0) - 1.0)
    return f"{'+' if hz >= 0 else ''}{int(round(hz))}Hz"


def merge_arc(points: Iterable[EmotionPoint], clamp_len: int = 0) -> EmotionPoint:
    """把一串情感点合成一个（加权平均，强度大的权重高）。"""
    pts = list(points)
    if not pts:
        return EmotionPoint()
    if clamp_len and len(pts) > clamp_len:
        pts = pts[:clamp_len]
    wsum = sum((0.2 + p.intensity) * (0.5 + abs(p.valence)) for p in pts)
    w = [(0.2 + p.intensity) * (0.5 + abs(p.valence)) for p in pts]
    v = sum(p.valence * k for p, k in zip(pts, w)) / wsum
    a = sum(p.arousal * k for p, k in zip(pts, w)) / wsum
    d = sum(p.dominance * k for p, k in zip(pts, w)) / wsum
    i = max(p.intensity for p in pts)
    # 标签取「强度 × 效价绝对值」最大的那一句（不是单纯最激动的那句）
    top = max(zip(pts, w), key=lambda pw: pw[1])[0]
    return EmotionPoint(v, a, d, i, top.label, top.confidence, source="arc").normalized()