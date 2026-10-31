"""小洛语音情感层 —— 纯标准库，零依赖。

    from emotion import render, analyze
    p = analyze("今天好累啊…")
    r = render(text, synth_fn)          # synth_fn(text, overrides) -> wav bytes

模块：
    schema     数据结构（EmotionPoint / ProsodyPlan）
    lexicon    情感词典
    analyzer   文本 → 情感（规则 + 情绪惯性 + 可选 LLM）
    planner    情感 → 韵律参数（曲线/颤音/气声/停顿/拖长）
    dsp        纯 Python PCM 处理（变调/气声/拖尾/限幅）
    metrics    客观情感指纹（F0/能量/停顿率/音程熵）
    renderer   整条渲染链路
    persona    人设偏置（小洛）
    llm        可选 LLM 结构化标注
    api        本地 HTTP 服务（默认 9887）
    cli        命令行
"""
from __future__ import annotations

from .schema import EmotionPoint, ProsodyPlan, SegmentPlan
from .analyzer import analyze, score_text, split_clauses, EmotionState
from .planner import build_plan, plan_segment
from .renderer import render, RenderResult, TooLong
from . import dsp, metrics, persona

__all__ = [
    "EmotionPoint", "ProsodyPlan", "SegmentPlan",
    "analyze", "score_text", "split_clauses", "EmotionState",
    "build_plan", "plan_segment", "render", "RenderResult", "TooLong",
    "dsp", "metrics", "persona",
]