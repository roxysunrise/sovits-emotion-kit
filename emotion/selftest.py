"""selftest —— 不联网也能验证「情感真的做出来了」。

验证四件事：
  1. WAV 读写与限幅没问题
  2. 变调真的变了（正弦 200Hz → +2 半音 ≈ 224.5Hz）
  3. 分析器方向对（开心 V/A 高、难过 V/A 低、生气 D 高、委屈 D 低）
  4. 同一条假语音，用不同情绪渲染后，客观指纹按期望方向移动

跑法：python -m emotion.selftest
"""
from __future__ import annotations

import json
import math
import os
import random
import sys

from . import analyzer as A
from . import dsp
from . import metrics
from . import planner as PL
from . import persona as P
from . import renderer as R
from .schema import EmotionPoint

RATE = 24000
OK = "[PASS]"
NG = "[FAIL]"
_rows: list[tuple[bool, str, str]] = []


def check(name: str, cond: bool, detail: str = "") -> bool:
    _rows.append((bool(cond), name, detail))
    print(f"{OK if cond else NG} {name}" + (f"  {detail}" if detail else ""))
    return bool(cond)


def tone(freq: float, sec: float, rate: int = RATE) -> list[float]:
    n = int(rate * sec)
    return [0.4 * math.sin(2 * math.pi * freq * i / rate) for i in range(n)]


def fake_speech(sec: float = 2.2, f0: float = 200.0, rate: int = RATE) -> list[float]:
    """假语音：基频 + 谐波 + 音节包络 + 停顿，用来测整条 DSP 链。"""
    n = int(rate * sec)
    rnd = random.Random(3)
    out = [0.0] * n
    for i in range(n):
        t = i / rate
        # 音节包络：每 0.22s 一个音节，带一点随机
        syl = 0.5 + 0.5 * math.sin(2 * math.pi * t / 0.22 + rnd.random() * 0.05)
        env = max(0.0, syl) ** 1.6
        if 0.9 < t < 1.05 or 1.6 < t < 1.7:      # 两处停顿
            env *= 0.05
        # 基频微滑
        f = f0 * (1.0 + 0.03 * math.sin(2 * math.pi * 0.7 * t))
        ph = 2 * math.pi * f * t
        s = math.sin(ph) + 0.5 * math.sin(2 * ph) + 0.25 * math.sin(3 * ph)
        out[i] = 0.22 * s * env
    return out


def estimate_f0(x: list[float], rate: int = RATE) -> float:
    fp = metrics.fingerprint(x, rate)
    return fp.get("f0_med", 0.0)


# ───────────────────────── 用例 ─────────────────────────

def test_wav() -> None:
    x = tone(200.0, 0.5)
    wav = dsp.build_wav(x, RATE)
    y, rate = dsp.parse_wav(wav)
    check("WAV 往返", rate == RATE and abs(len(x) - len(y)) <= 1 and len(wav) > 44,
          f"{len(wav)}B {rate}Hz")
    loud = dsp.normalize_rms([v * 40 for v in x], -19.0)
    check("软限幅不超 1.0", max(abs(v) for v in loud) <= 1.0,
          f"peak={max(abs(v) for v in loud):.3f}")


def test_pitch() -> None:
    x = tone(200.0, 1.0)
    base = estimate_f0(x)
    up = dsp.pitch_contour(x, RATE, base_semitones=2.0)
    f_up = estimate_f0(up)
    down = dsp.pitch_contour(x, RATE, base_semitones=-2.0)
    f_down = estimate_f0(down)
    ratio_up = f_up / base if base else 0
    ratio_dn = f_down / base if base else 0
    check("变调 +2 半音 ≈ ×1.122", abs(ratio_up - 1.1225) < 0.05,
          f"{base:.1f}→{f_up:.1f}Hz (×{ratio_up:.3f})")
    check("变调 -2 半音 ≈ ×0.891", abs(ratio_dn - 0.8909) < 0.05,
          f"{base:.1f}→{f_down:.1f}Hz (×{ratio_dn:.3f})")
    check("变调后时长不变", abs(len(up) - len(x)) <= 2, f"{len(x)}→{len(up)}")


def test_analyzer() -> None:
    cases = [
        ("今天好开心呀！谢谢你陪我等这么久～", "开心", "v_pos"),
        ("呜呜…我好难过，别走好不好", "难过", "v_neg"),
        ("你烦不烦啊！气死我了！！", "生气", "a_high"),
        ("人家想你了嘛，抱抱好不好～", "撒娇", "v_pos"),
        ("好累啊…想睡了", "疲惫", "a_low"),
    ]
    got = {}
    for text, want, _tag in cases:
        pt = A.analyze(text)
        got[want] = pt
        print(f"      「{text[:16]}…」→ {pt.label} V={pt.valence:+.2f} "
              f"A={pt.arousal:.2f} D={pt.dominance:.2f} I={pt.intensity:.2f}")
    check("开心 → V>0.3", got["开心"].valence > 0.3, f"V={got['开心'].valence:+.2f}")
    check("难过 → V<-0.2", got["难过"].valence < -0.2, f"V={got['难过'].valence:+.2f}")
    check("生气 → A>0.5 且 D>0.5",
          got["生气"].arousal > 0.5 and got["生气"].dominance > 0.5,
          f"A={got['生气'].arousal:.2f} D={got['生气'].dominance:.2f}")
    check("撒娇 → D<0.45", got["撒娇"].dominance < 0.45, f"D={got['撒娇'].dominance:.2f}")
    check("疲惫 → A<0.45", got["疲惫"].arousal < 0.45, f"A={got['疲惫'].arousal:.2f}")
    # 情绪弧：一句话里前半低落后半扬起
    arc_text = "今天真的好累啊…不过看到你就好啦！"
    pt = A.analyze(arc_text)
    check("逐句情感弧 ≥2 段", len(pt.arc) >= 2, f"{len(pt.arc)} 段")
    if len(pt.arc) >= 2:
        check("弧线后段比前段亮", pt.arc[-1].valence > pt.arc[0].valence,
              f"{pt.arc[0].valence:+.2f} → {pt.arc[-1].valence:+.2f}")


def test_planner() -> None:
    p = P.load()
    calm = EmotionPoint(0.1, 0.2, 0.5, 0.2, "平静", 0.6)
    happy = EmotionPoint(0.8, 0.85, 0.6, 0.85, "兴奋", 0.8)
    sad = EmotionPoint(-0.7, 0.15, 0.3, 0.8, "难过", 0.8)
    s_c = PL.plan_segment("测试", calm, p)
    s_h = PL.plan_segment("测试", happy, p)
    s_s = PL.plan_segment("测试", sad, p)
    check("兴奋比平静更快更高", s_h.rate_pct > s_c.rate_pct and s_h.base_semitones > s_c.base_semitones,
          f"rate {s_c.rate_pct}→{s_h.rate_pct}, st {s_c.base_semitones}→{s_h.base_semitones}")
    check("难过比平静更慢更低", s_s.rate_pct < s_c.rate_pct and s_s.base_semitones < s_c.base_semitones,
          f"rate {s_c.rate_pct}→{s_s.rate_pct}, st {s_c.base_semitones}→{s_s.base_semitones}")
    check("难过停顿更长", s_s.pause_after > s_c.pause_after,
          f"{s_c.pause_after}s→{s_s.pause_after}s")
    check("难过有句尾下沉", s_s.contour[-1][1] < s_s.contour[0][1],
          f"{s_s.contour[0][1]}→{s_s.contour[-1][1]}")
    check("兴奋颤音更强", s_h.vibrato_st > s_c.vibrato_st,
          f"{s_c.vibrato_st}→{s_h.vibrato_st}")


def test_render_direction() -> None:
    """同一条假语音，按不同情绪渲染 → 指纹方向必须对。"""
    p = P.load()

    def make_synth(seed: int):
        def synth(text: str, overrides: dict) -> bytes:
            x = fake_speech(sec=1.8, f0=200.0)
            return dsp.build_wav(x, RATE)
        return synth

    base_wav, _ = _render_with("今天就先这样吧，没什么特别的。", p, make_synth(1), EmotionPoint(0.1, 0.22, 0.5, 0.2, "平静", 0.6))
    fp_base = metrics.fingerprint_wav(base_wav)

    targets = [
        ("兴奋", "啊啊啊太好了！我超开心的！！！", EmotionPoint(0.88, 0.9, 0.65, 0.9, "兴奋", 0.85)),
        ("难过", "呜…好难过，我有点撑不住了……", EmotionPoint(-0.75, 0.15, 0.25, 0.85, "难过", 0.85)),
        ("生气", "你烦不烦啊！别再说了！！", EmotionPoint(-0.6, 0.85, 0.85, 0.9, "生气", 0.85)),
    ]
    for label, text, pt in targets:
        wav, meta = _render_with(text, p, make_synth(2), pt)
        fp = metrics.fingerprint_wav(wav)
        rows = metrics.check_direction(fp_base, fp, label)
        passed = [r for r in rows if r["passed"]]
        detail = " ".join(f"{r['metric']}{r['delta']:+.2f}" for r in rows)
        check(f"{label} 情感方向", len(passed) >= max(1, len(rows) - 1),
              f"{len(passed)}/{len(rows)} {detail}")

    # 气声：温柔 vs 生气，安静段过零率应该更高
    soft_wav, _ = _render_with("早点睡，别熬夜了，我陪着你。", p, make_synth(3),
                               EmotionPoint(0.55, 0.26, 0.45, 0.7, "温柔", 0.8))
    fp_soft = metrics.fingerprint_wav(soft_wav)
    check("温柔有气声", fp_soft["hf_ratio"] > fp_base["hf_ratio"] + 0.02,
          f"hf_ratio {fp_base['hf_ratio']}→{fp_soft['hf_ratio']}")


def _render_with(text: str, p: dict, synth, prior: EmotionPoint) -> tuple[bytes, dict]:
    """直接给定期望情感（跳过分析器），专测「情感 → 音频」这段有没有真的生效。"""
    r = R.render(text, synth, persona=p, emotion=prior, with_metrics=True,
                 log=lambda *_: None)
    return r.wav, r.meta


def test_full_chain() -> None:
    """整段（多小句）渲染：时长、长度上限、WAV 合规。"""
    p = P.load()
    synth = lambda text, ov: dsp.build_wav(fake_speech(sec=1.2, f0=205.0), RATE)
    text = ("今天真的好累啊…不过没关系啦，看到你就好啦！"
            "等下要不要一起听歌？人家想听你唱歌嘛～")
    r = R.render(text, synth, persona=p, with_metrics=True, log=lambda *_: None)
    x, rate = dsp.parse_wav(r.wav)
    dur = len(x) / rate
    check("整段渲染出时长", 0.5 < dur <= 58.0, f"{dur:.2f}s / {len(r.meta['segments'])} 小句")
    check("元数据含情感", "emotion" in r.meta and r.meta["emotion"]["label"],
          r.meta["emotion"]["label"])
    check("小句数 = 分段数", len(r.meta["segments"]) == len(r.meta["plan"]["segments"]),
          f"{len(r.meta['segments'])}")
    peak = max(abs(v) for v in x)
    check("峰值不爆", peak <= 0.985, f"peak={peak:.3f}")


def test_emoji_off() -> None:
    """小洛不用 emoji / 表情包：文本要在送后端之前被洗干净，且不吃表情的情绪票。"""
    from . import lexicon as L

    out, n = L.strip_emoji("哈哈😂😂 真的吗🤔?")
    check("emoji 被剥掉", "😂" not in out and "🤔" not in out, f"{n} 个 → {out!r}")

    clean, cnt = L.clean_for_speech("我好难过😭 [表情包:大哭] [图片] 别走")
    check("标记与 emoji 都清掉", "[" not in clean and "😭" not in clean,
          f"{cnt} → {clean!r}")

    pt_emoji = A.analyze("我好难过😭😭😭")
    pt_plain = A.analyze("我好难过")
    check("emoji 不再影响情感判定", pt_emoji.label == pt_plain.label,
          f"{pt_emoji.label} / {pt_plain.label}")

    pt_pure = A.analyze("😭😭😭🥺")
    check("纯 emoji 不判成难过", pt_pure.label != "哭腔" and pt_pure.label != "委屈",
          f"{pt_pure.label}")

    # 整条渲染：送后端的文本里不能再有 emoji
    p = P.load()
    synth = lambda text, ov: dsp.build_wav(fake_speech(sec=1.0, f0=200.0), RATE)
    r = R.render("呜呜好难过😭😭 [表情包:大哭]", synth, persona=p, with_metrics=False,
                 log=lambda *_: None)
    segs = [s["text"] for s in r.meta["segments"]]
    check("渲染送后端文本无 emoji", all("😭" not in s and "[" not in s for s in segs),
          f"{segs} / cleaned={r.meta['cleaned']}")


def test_gsv_refs() -> None:
    """SoVITS 侧的情感：ref 库 + 参数映射 + hint 优先级（不需要 9880 在跑）。"""
    from . import refs

    hint_pt = A.analyze("今日は嬉しいことがあったの。", hint_label="难过")
    check("hint 压住文本情绪（难过 vs 文本里的嬉しい）", hint_pt.label == "难过",
          f"{hint_pt.label} / src={hint_pt.source}")
    hint_pt2 = A.analyze("今日は嬉しいことがあったの。", hint_label="生气")
    check("hint 换一个标签也跟着换", hint_pt2.label == "生气", hint_pt2.label)

    ov_sad = refs.pick("难过", 0.6, out_dir=refs.DEFAULT_DIR)
    ov_happy = refs.pick("开心", 0.6, out_dir=refs.DEFAULT_DIR)
    have_refs = bool(ov_sad and ov_happy)
    check("情绪 → ref 参数", have_refs and "难过" in str(ov_sad.get("gpt_sovits_ref_audio"))
          and "开心" in str(ov_happy.get("gpt_sovits_ref_audio")),
          f"{ov_sad.get('gpt_sovits_ref_audio', '')} | {ov_happy.get('gpt_sovits_ref_audio', '')}")
    check("难过 比开心 更慢更保守",
          have_refs and float(ov_sad["gpt_sovits_speed"]) < float(ov_happy["gpt_sovits_speed"])
          and float(ov_sad["gpt_sovits_temperature"]) < float(ov_happy["gpt_sovits_temperature"]),
          f"speed {ov_sad.get('gpt_sovits_speed')} vs {ov_happy.get('gpt_sovits_speed')}")
    check("未知标签落回平静", refs.proto("莫名其妙") == "平静")

    ok, info = refs.validate(os.path.join(refs.DEFAULT_DIR, "平静.wav"))
    check("ref 校验通过（3~12s / ≥16k / 非静音）", ok, json.dumps(info, ensure_ascii=False)[:90])


def test_wsola() -> None:
    """变调不能带"回音/重音"：长度保持、无重复段、调制不失控。"""
    from . import refs as _refs

    x, rate = dsp.parse_wav(dsp.build_wav(tone(220.0, 1.2), RATE))
    y = dsp.pitch_shift_wsola(x, rate, -2.0)
    check("WSOLA 变调后时长不变", abs(len(y) - len(x)) <= max(8, len(x) // 200),
          f"{len(x)} → {len(y)} 样本")
    check("WSOLA 变调音高变低", metrics.fingerprint(y, rate)["f0_med"] <
          metrics.fingerprint(x, rate)["f0_med"] - 8,
          f"{metrics.fingerprint(x, rate)['f0_med']:.1f} → "
          f"{metrics.fingerprint(y, rate)['f0_med']:.1f}Hz")

    # 回音检查：拿真实 ref（有的话）做变调，自相关峰不应翻倍
    base = os.path.join(_refs.DEFAULT_DIR, "平静.wav")
    if not os.path.exists(base):
        return
    xb, rb = dsp.parse_wav(open(base, "rb").read())

    def echo(sig: list[float], r: int) -> float:
        lag_min, lag_max = int(r * 0.02), min(int(r * 0.30), len(sig) // 2)
        e0 = sum(v * v for v in sig) or 1e-9
        best, lag = 0.0, lag_min
        while lag <= lag_max:
            s = 0.0
            i = 0
            while i < len(sig) - lag:
                s += sig[i] * sig[i + lag]
                i += 4
            best = max(best, abs(s / e0) / max(1e-9, (len(sig) - lag) / len(sig)))
            lag += max(1, int(r * 0.002))
        return best

    yb = dsp.pitch_shift_wsola(xb, rb, -2.0)
    e0, e1 = echo(xb, rb), echo(yb, rb)
    check("变调不引入重复段（回音不翻倍）", e1 <= e0 * 2.2 + 0.005,
          f"原始 {e0:.3f} → 变调后 {e1:.3f}")


def main() -> int:
    print("=" * 62)
    print("  小洛语音情感层 自检（纯标准库，不联网）")
    print("=" * 62)
    test_wav()
    test_pitch()
    test_analyzer()
    test_planner()
    test_render_direction()
    test_full_chain()
    test_emoji_off()
    test_gsv_refs()
    test_wsola()
    bad = [r for r in _rows if not r[0]]
    print("-" * 62)
    print(f"  合计 {len(_rows)} 项，通过 {len(_rows) - len(bad)}，失败 {len(bad)}")
    for _ok, name, detail in bad:
        print(f"    ✗ {name} {detail}")
    print("=" * 62)
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())