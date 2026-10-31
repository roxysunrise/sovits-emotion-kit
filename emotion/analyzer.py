"""情感分析器 —— 文本 → EmotionPoint（含逐句情感弧）。

为什么不用「离散标签分类」：
    QQ 里一句话的情绪是连续变化的。小洛说「今天好累啊…不过看到你就好啦」，
    前半句该往下沉、后半句该扬起来。所以分析器做两件事：
      1) 整句汇总（给后端定基调）
      2) 逐小句拆解（给 DSP 做句内情绪走向）

打分方式是「词典命中 + 修饰词 + 标点 + 表情 + 内联表演指令」的加权投票，
不在词典里的情绪词不会失控——没有命中就回到基线（平静），confidence 会掉。

可选 LLM 精标（emotion/llm.py）：LLM 返回结构化 JSON 时以它为主，规则做兜底和校验。
"""
from __future__ import annotations

import json
import os
import re

from . import lexicon as L
from .schema import EmotionPoint, merge_arc

BASELINE = EmotionPoint(0.12, 0.24, 0.50, 0.22, "平静", 0.55, source="baseline")

# 小洛的基线：偏甜、偏轻快一点（人设基线，不是情绪）
XIAOLUO_BASELINE = EmotionPoint(0.20, 0.30, 0.45, 0.25, "平静", 0.55, source="persona")

RE_INLINE = re.compile(r"[（(【\[<](.{1,12}?)[）)】\]>]")

# 句读切分：中文标点 + 空白，保证小句不超过 ~24 字
RE_SPLIT = re.compile(r"(?<=[。！？!?；;…~\n])|(?<=[，,])")


def _alias(label: str) -> str:
    key = label.strip().lower()
    if key in L.LABEL_ALIASES:
        return L.LABEL_ALIASES[key]
    return label.strip()


def extract_hints(text: str) -> tuple[str, list[str]]:
    """抽出内联表演指令，返回 (干净文本, 标签列表)。"""
    labels: list[str] = []
    for m in RE_INLINE.finditer(text):
        inner = m.group(1)
        norm = inner.replace("emotion=", "").replace("语气:", "").replace("语气：", "")
        lbl = _alias(norm)
        if lbl in L.LABEL_PRIORS:
            labels.append(lbl)
            continue
        for token, target in L.INLINE_HINTS.items():
            if token in norm:
                labels.append(target)
                break
    clean = RE_INLINE.sub("", text).strip()
    return clean, labels


def split_clauses(text: str, max_len: int = 26) -> list[str]:
    """切小句。太长的再按逗号/顿号二次切，避免一个段落只有一种情绪。"""
    text = text.strip()
    if not text:
        return []
    raw = [s for s in RE_SPLIT.split(text) if s and s.strip()]
    out: list[str] = []
    for part in raw:
        part = part.strip()
        if len(part) <= max_len:
            out.append(part)
            continue
        buf = ""
        for ch in part:
            buf += ch
            if len(buf) >= max_len and ch in "、，,和跟与":
                out.append(buf)
                buf = ""
        if buf:
            out.append(buf)
    return out or [text]


# ───────────────────────── 单句打分 ─────────────────────────

def score_text(text: str, prior: EmotionPoint | None = None, use_emoji: bool = False) -> EmotionPoint:
    """一句话 → EmotionPoint。规则版。use_emoji=False 时表情不参与打分。"""
    base = prior or BASELINE
    clean, hint_labels = extract_hints(text)
    low = clean.lower()

    scores: dict[str, float] = {k: 0.0 for k in L.LABEL_PRIORS}
    # 1) 词典命中（长词优先，命中一次计一次，防止「哈哈哈哈」被算 4 次）
    matched_spans: list[tuple[int, int]] = []
    for label, words in L.WORDS.items():
        for w in sorted(words, key=len, reverse=True):
            wl = w.lower()
            if not wl:
                continue
            start = 0
            while True:
                idx = low.find(wl, start)
                if idx < 0:
                    break
                end = idx + len(wl)
                if any(a <= idx < b or a < end <= b for a, b in matched_spans):
                    start = end
                    continue
                matched_spans.append((idx, end))
                # 长词权重更高
                scores[label] += 1.0 + 0.35 * (len(w) - 1)
                start = end

    # 2) 内联表演指令：权重很高（是显式要求）
    for lbl in hint_labels:
        scores[lbl] = scores.get(lbl, 0.0) + 2.2

    # 3) 表情（2026-10-31 起小洛不用 emoji，默认不吃这一票，免得语气被表情带偏）
    if use_emoji:
        for ch in clean:
            lbl = L.EMOJI.get(ch)
            if lbl:
                scores[lbl] += 1.4

    # 4) 语气词微调
    dv = da = dd = 0.0
    for p, (pv, pa, pd, pl) in L.PARTICLES.items():
        cnt = clean.count(p)
        if not cnt:
            continue
        cnt = min(cnt, 3)
        dv += pv * cnt
        da += pa * cnt
        dd += pd * cnt
        if pl in scores:
            scores[pl] += 0.20 * cnt

    total = sum(scores.values())
    if total <= 0:
        # 没有命中：回到基线，标点还能给一点情绪
        da_extra, dv_extra, di = _punct_effect(clean)
        return EmotionPoint(
            base.valence + dv + dv_extra,
            base.arousal + da + da_extra,
            base.dominance + dd,
            max(0.08, base.intensity * 0.65 + di),
            base.label, 0.35, source="baseline",
        ).normalized()

    v = a = d = 0.0
    for lbl, s in scores.items():
        if s <= 0:
            continue
        pv, pa, pd = L.LABEL_PRIORS[lbl]
        v += pv * s
        a += pa * s
        d += pd * s
    v, a, d = v / total, a / total, d / total

    # 5) 标点
    da_extra, dv_extra, di = _punct_effect(clean)
    a += da_extra
    v += dv_extra

    # 6) 程度副词 / 弱化 / 否定
    amp = 0.0
    for w, delta in L.AMPLIFIERS.items():
        if w in clean:
            amp += delta
    damp = 0.0
    for w, delta in L.DAMPENERS.items():
        if w in clean:
            damp += delta
    intensity = 0.28 + 0.42 * min(1.0, total / 3.0) + amp + damp + di
    intensity = max(0.05, min(1.0, intensity))

    # 否定：只有「否定词 + 情绪词」贴在一起才算否定（「不过」「不错」不算）
    neg_hits = _negation_hits(clean, matched_spans)
    if neg_hits and v > 0:
        v = -v * min(0.55 + 0.1 * neg_hits, 0.8)
    elif neg_hits and v < 0:
        v = v * 0.45

    # 强度不足时向基线收敛（防止一个词就把语气带飞）
    if intensity < 0.35:
        v = base.valence * 0.5 + v * 0.5
        a = base.arousal * 0.5 + a * 0.5

    label, margin = _pick_label(scores, v, a, d)
    label = reconcile(label, v, a, d)
    conf = min(0.95, 0.42 + 0.35 * min(1.0, total / 3.0) + 0.5 * margin)

    return EmotionPoint(v, a, d, intensity, label, conf, source="rule").normalized()


def _negation_hits(text: str, spans: list[tuple[int, int]], window: int = 4) -> int:
    """数「真正起作用」的否定：否定词后面 window 个字符内出现了命中的情绪词。"""
    hits = 0
    for neg in L.NEGATORS:
        start = 0
        while True:
            pos = text.find(neg, start)
            if pos < 0:
                break
            start = pos + len(neg)
            # 「不过/不错/不得了」这类例外词跳过
            if any(text.startswith(ex, pos) for ex in L.NEG_EXCEPT):
                continue
            lo, hi = pos + len(neg), pos + len(neg) + window
            if any(lo <= a < hi for a, _b in spans):
                hits += 1
    return hits


def reconcile(label: str, v: float, a: float, d: float) -> str:
    """标签与效价对账：不允许「撒娇」配哭泣，也不允许「生气」配甜。"""
    if v < -0.15 and label in L.POS_ONLY:
        if a < 0.32:
            return "难过"
        if d < 0.34:
            return "委屈"
        return "生气"
    if v > 0.30 and label in L.NEG_ONLY:
        return "温柔" if a < 0.42 else "开心"
    return label


def _punct_effect(text: str) -> tuple[float, float, float]:
    da = dv = di = 0.0
    for p, (pa, pv, pi) in L.PUNCT.items():
        c = text.count(p)
        if not c:
            continue
        c = min(c, 3)
        da += pa * c
        dv += pv * c
        di += pi * c
    # 感叹号堆积（！！！）再加码
    if re.search(r"[！!]{2,}", text):
        da += 0.10
        di += 0.06
    # 破折号/省略号延长 = 犹豫、拖音
    if re.search(r"[…]{2,}|\.{3,}", text):
        di += 0.04
        da -= 0.04
    # 重复字符（哈哈哈哈 / 呜呜呜 / 好好好）→ 强度++
    if re.search(r"(.)\1{2,}", text):
        di += 0.08
        da += 0.05
    return da, dv, di


def _pick_label(scores: dict[str, float], v: float, a: float, d: float) -> tuple[str, float]:
    if not scores or sum(scores.values()) <= 0:
        return "平静", 0.0
    ranked = sorted(((s, k) for k, s in scores.items() if s > 0), reverse=True)
    top_s, top_k = ranked[0]
    second = ranked[1][0] if len(ranked) > 1 else 0.0
    margin = (top_s - second) / max(top_s, 1e-6)

    # 词典命中但坐标明显偏其他象限时做一次矫正（例如「不生气」）
    if v < -0.25 and a < 0.30 and d < 0.45 and top_k not in ("难过", "疲惫", "冷淡"):
        top_k = "难过"
    if v < -0.25 and a > 0.62 and top_k not in ("生气", "害怕", "急切"):
        top_k = "生气"
    if v > 0.45 and a < 0.35 and d < 0.45:
        top_k = "温柔" if scores.get("温柔", 0) > 0 else "撒娇"
    return top_k, max(0.0, margin)


# ───────────────────────── 整段分析 ─────────────────────────

def analyze(
    text: str,
    prior: EmotionPoint | None = None,
    hint_label: str | None = None,
    use_llm: dict | None = None,
    use_emoji: bool = False,
) -> EmotionPoint:
    """整段文本 → EmotionPoint（带 arc）。

    prior       : 上一轮的继续情绪（情绪惯性），默认小洛基线
    hint_label  : 桥接/agent 直接给的情绪标签
    use_llm     : 传入 llm 配置 dict 则尝试 LLM 精标（见 emotion/llm.py）
    use_emoji   : 是否让表情参与打分。默认 False —— 小洛不用 emoji，
                  而且上游漏过来的 emoji 会把语气带偏（见 lexicon.clean_for_speech）
    """
    base = prior or XIAOLUO_BASELINE
    clean, hint_labels = extract_hints(text or "")
    clean, _cleaned = L.clean_for_speech(clean, keep_emoji=use_emoji)
    if not clean:
        return EmotionPoint(base.valence, base.arousal, base.dominance,
                            0.10, base.label, 0.4, source="empty").normalized()

    hint_pt = None
    if hint_label:
        lbl = _alias(hint_label)
        if lbl in L.LABEL_PRIORS:
            pv, pa, pd = L.LABEL_PRIORS[lbl]
            hint_pt = EmotionPoint(pv, pa, pd, 0.75, lbl, 0.9, source="hint").normalized()
            # 桥接/agent 给的是**明确指令**，不是参考意见：它必须压住文本基线。
            # 以前这里只做 0.55 的软融合，结果「hint=难过 + 文本里有 嬉しい」时
            # 判定会被文本翻成开心，「按情绪换 ref」就永远换不到想换的那个。
            base = hint_pt.lerp(base, 0.25)

    clauses = split_clauses(clean)
    arc: list[EmotionPoint] = []
    for c in clauses:
        pt = score_text(c, base)
        # 转折：后一句被前半句的负面压着会显得假，这里把效价抬回来
        head = c[:4]
        if any(head.startswith(w) or w in head for w in L.CONTRAST):
            pt = EmotionPoint(
                min(1.0, pt.valence + 0.22), min(1.0, pt.arousal + 0.06),
                pt.dominance, pt.intensity, pt.label, pt.confidence,
            ).normalized()
            pt.label = reconcile(pt.label, pt.valence, pt.arousal, pt.dominance)
        arc.append(pt)
    for lbl in hint_labels:
        pv, pa, pd = L.LABEL_PRIORS[lbl]
        arc.append(EmotionPoint(pv, pa, pd, 0.8, lbl, 0.9, source="hint").normalized())

    whole = merge_arc(arc if arc else [score_text(clean, base)])
    whole.arc = arc
    whole.label = reconcile(whole.label, whole.valence, whole.arousal, whole.dominance)

    # 情绪惯性：新情绪占大头，但强度越高越不让基线把情绪拉平
    alpha = 0.55 + 0.35 * whole.intensity
    if abs(whole.valence) > 0.35:
        alpha = max(alpha, 0.80)
    blended = base.lerp(whole, alpha)
    blended.label = reconcile(whole.label, blended.valence, blended.arousal, blended.dominance)
    blended.arc = whole.arc
    blended.source = "rule+state"

    if hint_pt is not None:
        # 标签以 hint 为准（它是上游的明确意图），强度/弧线仍让文本说话。
        # 这样「高唤醒的难过」「低唤醒的开心」都能表达出来，而不是被拉回平均。
        blended = hint_pt.lerp(blended, 0.30)
        blended.label = hint_pt.label
        blended.intensity = max(hint_pt.intensity * 0.6, blended.intensity)
        blended.arc = whole.arc
        blended.source = "hint+rule"

    if use_llm:
        try:
            from . import llm as _llm
            got = _llm.annotate(clean, whole, use_llm)
            if got is not None:
                # LLM 给基调，规则保强度与弧线
                got.arc = whole.arc
                got.intensity = max(got.intensity * 0.7, whole.intensity * 0.8)
                blended = blended.lerp(got, 0.7)
                blended.arc = whole.arc
                blended.source = "llm+rule"
        except Exception as e:  # LLM 挂了不影响出声
            blended.source = f"rule+state(llm:{type(e).__name__})"
    return blended.normalized()


# ───────────────────────── 情绪状态机 ─────────────────────────

class EmotionState:
    """跨轮情绪惯性：放在 profile 里，同一用户连续对话时情绪是连贯的。

    carry   : 上一轮残留比例（0.35 表示上一轮情绪还剩 35%）
    reset_s : 静默多久后情绪完全回到基线（秒）
    """

    def __init__(self, path: str | None = None, carry: float = 0.35, reset_s: float = 600.0):
        self.path = path
        self.carry = max(0.0, min(0.9, carry))
        self.reset_s = reset_s
        self._data: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not self.path or not os.path.exists(self.path):
            return
        try:
            with open(self.path, encoding="utf-8-sig") as f:
                self._data = json.load(f)
        except Exception:
            self._data = {}

    def _save(self) -> None:
        if not self.path:
            return
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=1)
            os.replace(tmp, self.path)
        except Exception:
            pass

    def prior(self, user: str, now: float) -> EmotionPoint:
        rec = self._data.get(user or "default")
        if not rec:
            return XIAOLUO_BASELINE
        try:
            age = max(0.0, now - float(rec.get("ts", 0)))
            pt = EmotionPoint.from_dict(rec["point"])
        except Exception:
            return XIAOLUO_BASELINE
        if age >= self.reset_s:
            return XIAOLUO_BASELINE
        # 时间越久，残留越少
        keep = self.carry * (1.0 - age / self.reset_s) ** 1.5
        return XIAOLUO_BASELINE.lerp(pt, keep)

    def update(self, user: str, point: EmotionPoint, now: float) -> None:
        self._data[user or "default"] = {"ts": now, "point": point.to_dict()}
        self._save()
