"""planner —— 情感点 → 韵律方案。

这里是全链路最关键的映射层。导出对话里说得很准：
    「情感不是靠调音量实现的」，靠三样：音高曲线、能量包络+呼吸感、时值/停顿。
所以每个小句输出的不是一个标量，而是一组**曲线与纹理**：

    contour      [(位置0~1, 半音偏移)]  句内音高走向（上扬/下沉/头顶一挑）
    vibrato_*    持续音的颤音（情绪越高、越满，颤得越明显）
    tremor       颤栗（哭腔/害怕专用，非周期抖动，做「抖」而不只是「抖音量」）
    breath       气声比例（温柔/撒娇/疲惫靠它，替换掉「加音量」这种蠢做法）
    pause_after  句后停顿（犹豫、哽咽靠它）
    tail_lengthen句尾拖长（软下来、舍不得、撒娇收尾）

标签只提供**倾向模板**，真正数值由 VAD 连续插值，所以「80% 的开心」也能表达。
"""
from __future__ import annotations

from .schema import EmotionPoint, ProsodyPlan, SegmentPlan, edge_hz, edge_pct
from .persona import DEFAULT_PERSONA

# 标签 → 情绪曲线形状（位置, 半音）
SHAPES: dict[str, list[tuple[float, float]]] = {
    "平静": [(0.0, -0.10), (0.5, 0.05), (1.0, -0.20)],
    "开心": [(0.0, -0.15), (0.35, 0.35), (0.75, 0.20), (1.0, 0.55)],
    "兴奋": [(0.0, 0.55), (0.3, 0.15), (0.6, 0.45), (1.0, 0.75)],
    "撒娇": [(0.0, -0.25), (0.55, 0.10), (1.0, 0.95)],
    "温柔": [(0.0, 0.10), (0.6, -0.05), (1.0, -0.35)],
    "感动": [(0.0, 0.05), (0.5, -0.15), (1.0, 0.25)],
    "难过": [(0.0, 0.10), (0.35, -0.25), (1.0, -1.05)],
    "委屈": [(0.0, 0.20), (0.4, -0.10), (1.0, -0.70)],
    "生气": [(0.0, 0.85), (0.25, 0.10), (0.7, 0.30), (1.0, -0.15)],
    "惊讶": [(0.0, 1.55), (0.18, 0.35), (0.6, 0.10), (1.0, 0.25)],
    "害羞": [(0.0, -0.30), (0.5, -0.10), (1.0, 0.35)],
    "疲惫": [(0.0, -0.35), (0.5, -0.55), (1.0, -1.25)],
    "害怕": [(0.0, 0.95), (0.4, 0.10), (1.0, -0.30)],
    "调皮": [(0.0, 0.10), (0.5, 0.45), (1.0, 1.10)],
    "冷淡": [(0.0, -0.15), (0.7, -0.25), (1.0, -0.45)],
    "急切": [(0.0, 0.30), (0.4, 0.55), (1.0, 0.35)],
    "哭腔": [(0.0, 0.35), (0.30, -0.35), (0.75, 0.10), (1.0, -0.95)],
}

# 柔软类情绪：会给更多气声、更长的句尾
SOFT = {"温柔", "撒娇", "害羞", "疲惫", "感动", "哭腔", "难过", "委屈"}
TENSE = {"兴奋", "生气", "急切", "害怕", "惊讶"}


def _lerp_shape(a: list[tuple[float, float]], b: list[tuple[float, float]],
                t: float) -> list[tuple[float, float]]:
    """两条曲线按 t 混合（采样到统一网格）。"""
    def sample(c: list[tuple[float, float]], x: float) -> float:
        if x <= c[0][0]:
            return c[0][1]
        for i in range(len(c) - 1):
            x0, y0 = c[i]
            x1, y1 = c[i + 1]
            if x0 <= x <= x1:
                k = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
                return y0 + (y1 - y0) * k
        return c[-1][1]

    grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    return [(g, sample(a, g) * (1 - t) + sample(b, g) * t) for g in grid]


def _shift(c: list[tuple[float, float]], dv: float) -> list[tuple[float, float]]:
    return [(x, y + dv) for x, y in c]


def _tail_rise(c: list[tuple[float, float]], amount: float) -> list[tuple[float, float]]:
    """人设的「句尾上扬倾向」：把最后 35% 抬起来。"""
    out = []
    for x, y in c:
        if x >= 0.65:
            w = (x - 0.65) / 0.35
            out.append((x, y + amount * w))
        else:
            out.append((x, y))
    return out


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def plan_segment(
    text: str,
    point: EmotionPoint,
    persona: dict | None = None,
    index: int = 0,
    total: int = 1,
    is_last: bool = True,
) -> SegmentPlan:
    """一个小句 → SegmentPlan。"""
    p = persona or DEFAULT_PERSONA
    bias = p.get("baseline", {})
    ebias = (p.get("emotion_bias") or {}).get(point.label, {})
    v, a, d, it = point.valence, point.arousal, point.dominance, point.intensity

    # ── 音高曲线：标签形状 → 按 arousal/valence/intensity 缩放 ──
    shape = SHAPES.get(point.label, SHAPES["平静"])
    base_shape = SHAPES["平静"]
    # 强度决定「像不像这个情绪」：强度低就往中性形状靠
    shape = _lerp_shape(base_shape, shape, _clamp(0.25 + 0.75 * it, 0.0, 1.0))
    # 效价拉低整体，唤醒度放大起伏
    amp = 0.55 + 0.85 * a + 0.35 * it
    shape = [(x, y * amp + (-0.45 * (1 - v) if v < 0 else 0.18 * v)) for x, y in shape]
    # 情绪弧：一句较长的话里，越靠后越贴这个情绪（前句留白）
    if total > 1 and not is_last:
        shape = [(x, y * 0.8) for x, y in shape]
    base_st = float(bias.get("semitones", 0.0)) + float(ebias.get("semitones", 0.0))
    base_st += 1.40 * (a - 0.30) - 0.90 * (1 - v) * 0.50
    base_st = _clamp(base_st, -2.5, 3.2)
    shape = _tail_rise(shape, float(bias.get("tail_rise", 0.0)) + float(ebias.get("tail_rise", 0.0)) * 0.5)

    # ── 语速 ──
    rate = float(bias.get("rate_pct", 0.0)) + float(ebias.get("rate_pct", 0.0))
    rate += 22.0 * (a - 0.3) - 14.0 * (1 - v) * 0.5 - 6.0 * (1 - it)
    if point.label in ("疲惫", "哭腔"):
        rate -= 6.0
    rate = _clamp(rate, -30.0, 40.0)

    # ── 颤音 / 颤栗 ──
    vib_hz = float(bias.get("vibrato_hz", 5.0)) + 1.7 * (a - 0.2) + 0.8 * it
    vib_st = float(bias.get("vibrato_st", 0.05)) + 0.30 * max(0.0, a - 0.45) + 0.12 * it
    if point.label in ("兴奋", "感动", "生气", "哭腔"):
        vib_st += 0.10
    vib_hz = _clamp(vib_hz, 3.0, 9.0)
    vib_st = _clamp(vib_st, 0.0, 0.45)

    tremor = 0.0
    if point.label in ("哭腔", "害怕", "委屈"):
        tremor = 0.06 + 0.16 * max(v == 0 and 0 or (1 - v) * 0.5, 0) + 0.10 * it
    elif point.label in ("疲惫", "害羞"):
        tremor = 0.03 + 0.05 * it
    tremor = _clamp(tremor, 0.0, 0.30)

    # ── 气声 ──
    breath = float(bias.get("breath", 0.10)) + float(ebias.get("breath", 0.0))
    if point.label in SOFT:
        breath += 0.10 + 0.18 * (1 - a) + 0.10 * it
    else:
        breath += 0.04 * (1 - a)
    breath = _clamp(breath, 0.0, 0.55)

    # ── 能量 ──
    gain_db = float(bias.get("energy_db", 0.0)) + 3.2 * (a - 0.35) - 2.4 * (1 - v) * 0.4
    gain_db += 0.8 * it - 0.6
    # 情绪必须真的「有音量差」，否则轻柔类永远显不出来
    if point.label in SOFT:
        gain_db -= 1.4 + 0.6 * it
    if point.label in TENSE:
        gain_db += 1.2 + 0.5 * it
    if point.label == "冷淡":
        gain_db -= 2.0
    gain_db = _clamp(gain_db, -8.0, 7.0)

    # ── 停顿 / 拖长 ──
    pause = 0.10 + 0.30 * (1 - a) + 0.18 * max(0.0, 1 - v) * 0.5 + 0.05 * it
    if point.label in ("疲惫", "难过", "哭腔"):
        pause += 0.10
    if point.label in ("急切", "兴奋", "生气"):
        pause *= 0.45
    pause = _clamp(pause, 0.06, 0.65)
    tail = 1.0 + 0.20 * max(0.0, 1 - a) * (0.5 + it)
    if point.label in ("撒娇", "难过", "哭腔", "疲惫", "温柔"):
        tail += 0.10
    tail = _clamp(tail, 1.0, 1.5)

    return SegmentPlan(
        text=text,
        rate_pct=round(rate, 2),
        base_semitones=round(base_st, 3),
        gain_db=round(gain_db, 2),
        contour=[(round(x, 3), round(y, 3)) for x, y in shape],
        vibrato_hz=round(vib_hz, 2),
        vibrato_st=round(vib_st, 3),
        tremor=round(tremor, 3),
        breath=round(breath, 3),
        pause_after=round(pause, 3),
        tail_lengthen=round(tail, 3),
        label=point.label,
        intensity=round(it, 3),
    )


def build_plan(
    clauses: list[str],
    whole: EmotionPoint,
    persona: dict | None = None,
    lang: str = "zh",
    base_f0: float = 220.0,
) -> ProsodyPlan:
    """整段 → ProsodyPlan。弧线按小句情感点走，没有弧线就按整句走。"""
    p = persona or DEFAULT_PERSONA
    arc = whole.arc or []
    segs: list[SegmentPlan] = []
    n = max(1, len(clauses))
    for i, c in enumerate(clauses):
        pt = arc[i] if i < len(arc) else whole
        # 句内弧线：整句情绪占 55%，小句情绪占 45%，避免过度抖动
        # 标签取小句自己的：以前写成 "pt.intensity >= whole.intensity 才用 pt.label"，
        # 而整句 intensity 基本总是 >= 小句的，于是**每个小句都被打回整句标签**
        # —— 长句的情绪转折就全丢了（实测 4 句全是"开心"）。
        # VAD 在这里已经按 55/45 混过，直接用 pt.label 不会抖。
        mixed = EmotionPoint(
            valence=whole.valence * 0.55 + pt.valence * 0.45,
            arousal=whole.arousal * 0.55 + pt.arousal * 0.45,
            dominance=whole.dominance * 0.55 + pt.dominance * 0.45,
            intensity=max(whole.intensity * 0.7, pt.intensity),
            label=pt.label if pt.label else whole.label,
            confidence=whole.confidence,
        ).normalized()
        segs.append(plan_segment(c, mixed, p, i, n, is_last=(i == n - 1)))

    # 后端汇总：edge 吃 rate/pitch 字符串，GSV 吃 speed_factor
    avg_rate = sum(s.rate_pct for s in segs) / max(1, len(segs))
    avg_st = sum(s.base_semitones for s in segs) / max(1, len(segs))
    backend = {
        "edge_rate": edge_pct(avg_rate),
        "edge_pitch": edge_hz(avg_st, base_f0),
        "gpt_sovits_speed": round(1.0 + avg_rate / 100.0, 3),
        "lang": lang,
    }
    notes = [
        f"emotion={whole.label} V={whole.valence:+.2f} A={whole.arousal:.2f} "
        f"D={whole.dominance:.2f} I={whole.intensity:.2f} conf={whole.confidence:.2f}",
        f"segments={len(segs)} rate={backend['edge_rate']} pitch={backend['edge_pitch']}",
    ]
    return ProsodyPlan(emotion=whole, segments=segs, backend=backend, notes=notes)