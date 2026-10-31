"""metrics —— 客观情感指纹，给「情感到底有没有做出来」一个可验证的答案。

导出对话里那篇 Phys. Rev. Research 6, 013136 的思路可以直接搬过来用：
    把演唱/语音当成「音高转移网络」，用转移分布的信息量（熵）刻画结构。
这里做轻量版，只取可算、可比的六个量：

    f0_med      基频中位数（Hz）        → 情绪高低（兴奋抬、疲惫沉）
    f0_sd_st    基频半音标准差           → 起伏幅度（激动抖得大）
    rms_db      响度（dBFS）            → 能量（生气/兴奋高）
    pause_ratio 静音帧占比               → 犹豫/哽咽（难过高）
    zcr_quiet   安静段过零率             → 气声比例（温柔/撒娇高）
    interval_entropy  音程转移熵（bit）   → 旋律/语调的信息量（越花越散）

用法：
    from emotion import metrics
    fp = metrics.fingerprint(wav_bytes)
    metrics.assert_direction(fp_calm, fp_sad, "难过")   # 断言方向对不对
"""
from __future__ import annotations

import math
import struct
import cmath

from . import dsp

# ───────────────────────── FFT ─────────────────────────

def _fft(a: list[complex]) -> None:
    """原地 radix-2 FFT（长度必须是 2 的幂）。"""
    n = len(a)
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j |= bit
        if i < j:
            a[i], a[j] = a[j], a[i]
    length = 2
    while length <= n:
        ang = -2 * math.pi / length
        wl = complex(math.cos(ang), math.sin(ang))
        for i in range(0, n, length):
            w = 1 + 0j
            half = length >> 1
            for k in range(i, i + half):
                u = a[k]
                v = a[k + half] * w
                a[k] = u + v
                a[k + half] = u - v
                w *= wl
        length <<= 1


def _decimate(x: list[float], factor: int) -> list[float]:
    if factor <= 1:
        return list(x)
    out = []
    for i in range(0, len(x) - factor + 1, factor):
        out.append(sum(x[i:i + factor]) / factor)
    return out


def f0_track(x: list[float], rate: int, fmin: float = 60.0, fmax: float = 500.0,
             frame_ms: float = 40.0, hop_ms: float = 20.0) -> tuple[list[float], list[float]]:
    """返回 (f0 序列 Hz，0 表示清音; 每帧 rms)。

用 YIN 的累积均值归一化差分（CMND）判周期，比裸自相关的最大值稳：
自相关的最大值有很强的倍频/半频倾向，女声会被判高一个八度或低一个八度。
自相关仍然用 FFT 算（纯 Python 只求够快），差分由 E0+Eτ-2r(τ) 得到。
"""
    dec = 2 if rate > 16000 else 1
    sr = rate / dec
    y = _decimate(x, dec)
    frame = int(sr * frame_ms / 1000.0)
    hop = int(sr * hop_ms / 1000.0)
    if frame < 64:
        frame = 64
    nfft = 1
    while nfft < frame * 2:
        nfft <<= 1
    win = [0.5 - 0.5 * math.cos(2 * math.pi * i / (frame - 1)) for i in range(frame)]
    lag_min = max(2, int(sr / fmax))
    lag_max = min(frame - 2, int(sr / fmin))
    # 注意：这里**不加窗**。YIN 的差分用未经加窗的原始样本算，能量项才对得上，
    # 加窗会让 d(τ) 出现负值被 clamp 成 0，CMND 立刻在 lag_min 处穿阈值（f0 全变成 fmax）。
    _ = win

    # 帧能量的前缀和（用于 Eτ）
    pre = [0.0] * (len(y) + 1)
    for i, v in enumerate(y):
        pre[i + 1] = pre[i] + v * v

    f0s: list[float] = []
    rmss: list[float] = []
    pos = 0
    while pos + frame <= len(y):
        seg = y[pos:pos + frame]
        rms = math.sqrt(sum(v * v for v in seg) / frame)
        rmss.append(rms)
        if rms < 1e-4:
            f0s.append(0.0)
            pos += hop
            continue

        buf = [complex(v, 0.0) for v in seg] + [0j] * (nfft - frame)
        _fft(buf)
        for i in range(nfft):
            buf[i] = (buf[i].real ** 2 + buf[i].imag ** 2) + 0j
        # 逆变换 = 共轭 FFT 再共轭
        for i in range(nfft):
            buf[i] = complex(buf[i].real, -buf[i].imag)
        _fft(buf)
        ac = [buf[i].real / nfft for i in range(nfft)]
        e0 = ac[0] if ac[0] > 1e-12 else 1e-12

        # YIN 差分 + 累积均值归一化
        d = [0.0] * (lag_max + 2)
        running = 0.0
        cmnd = [1.0] * (lag_max + 2)
        for tau in range(1, lag_max + 2):
            et = pre[pos + frame + tau] - pre[pos + tau] if pos + frame + tau <= len(y) else e0
            dv = e0 + et - 2.0 * ac[tau] if tau < len(ac) else e0
            d[tau] = max(0.0, dv)
            running += d[tau]
            cmnd[tau] = d[tau] * tau / running if running > 1e-12 else 1.0

        # 找第一个低于阈值的谷（找不到就退而求其次取全局最小）
        best = 0
        found = False
        tau = lag_min
        while tau <= lag_max:
            if cmnd[tau] < 0.16:
                while tau + 1 <= lag_max and cmnd[tau + 1] < cmnd[tau]:
                    tau += 1
                best = tau
                found = True
                break
            tau += 1
        if not found:
            best = min(range(lag_min, lag_max + 1), key=lambda t: cmnd[t])
            if cmnd[best] > 0.35:
                f0s.append(0.0)
                pos += hop
                continue

        if 1 <= best < lag_max + 1:
            a, b, c = cmnd[best - 1], cmnd[best], cmnd[best + 1]
            den = (a - 2 * b + c)
            delta = 0.5 * (a - c) / den if abs(den) > 1e-12 else 0.0
            lag_est = best + max(-0.5, min(0.5, delta))
        else:
            lag_est = float(best)
        f0s.append(sr / lag_est if lag_est else 0.0)
        pos += hop
    return f0s, rmss


def _zcr(x: list[float]) -> float:
    if len(x) < 2:
        return 0.0
    c = 0
    prev = x[0]
    for v in x[1:]:
        if (v >= 0) != (prev >= 0):
            c += 1
        prev = v
    return c / float(len(x) - 1)


def interval_entropy(f0s: list[float]) -> float:
    """音程转移熵（bit）：把连续浊音帧的半音差量化后做直方图。"""
    semis = [12.0 * math.log2(f / 100.0) for f in f0s if f > 0]
    if len(semis) < 4:
        return 0.0
    hist: dict[int, int] = {}
    total = 0
    for a, b in zip(semis, semis[1:]):
        d = int(round(b - a))
        if abs(d) > 12:
            continue
        hist[d] = hist.get(d, 0) + 1
        total += 1
    if not total:
        return 0.0
    ent = 0.0
    for c in hist.values():
        p = c / total
        ent -= p * math.log2(p)
    return ent


# ───────────────────────── 指纹 ─────────────────────────

def fingerprint(samples: list[float], rate: int) -> dict:
    if not samples:
        return {"duration": 0.0}
    f0s, rmss = f0_track(samples, rate)
    voiced = [f for f in f0s if f > 0]
    f0_sorted = sorted(voiced)
    f0_med = f0_sorted[len(f0_sorted) // 2] if f0_sorted else 0.0
    semis = [12.0 * math.log2(f / f0_med) for f in voiced] if f0_med else []
    if semis:
        mean = sum(semis) / len(semis)
        f0_sd = math.sqrt(sum((s - mean) ** 2 for s in semis) / len(semis))
    else:
        f0_sd = 0.0

    peak = max(rmss) if rmss else 0.0
    th = peak * 0.10 if peak > 0 else 1e-9
    # 停顿 = 所有低于阈值（含数字静音）的帧；气声只看「有信号但很轻」的帧
    silent = [i for i, r in enumerate(rmss) if r < th]
    quiet = [i for i in silent if rmss[i] > 1e-6]
    pause_ratio = len(silent) / float(len(rmss)) if rmss else 1.0

    fl = max(1, int(rate * 0.02))
    quiet_zcr = []
    quiet_energy = 0.0
    quiet_n = 0
    for i in quiet:
        seg = samples[i * fl:(i + 1) * fl]
        if seg:
            quiet_zcr.append(_zcr(seg))
            quiet_energy += sum(v * v for v in seg)
            quiet_n += len(seg)
    zcr_q = sum(quiet_zcr) / len(quiet_zcr) if quiet_zcr else 0.0
    quiet_rms = math.sqrt(quiet_energy / quiet_n) if quiet_n else 0.0
    quiet_rel_db = 20.0 * math.log10(max(quiet_rms, 1e-9) / max(peak, 1e-9)) if peak else -120.0

    # 气声/明亮度代理：一阶差分的能量 / 总能量。噪声占比越高这个比值越大
    # （气声、耳语、呼吸声都会把它抬起来，浊音越「闷」它越小）
    e_tot = sum(v * v for v in samples)
    d_tot = sum((samples[i + 1] - samples[i]) ** 2 for i in range(len(samples) - 1)) if len(samples) > 2 else 0.0
    hf_ratio = (d_tot / e_tot) if e_tot > 1e-12 else 0.0

    # 语速代理：能量包络过零次数 / 秒（音节密度）
    loud = [r for r in rmss]
    thr = peak * 0.25 if peak else 1e-9
    crossings = 0
    prev = False
    for r in loud:
        cur = r > thr
        if cur and not prev:
            crossings += 1
        prev = cur
    dur = len(samples) / float(rate)
    rate_syl = crossings / dur if dur > 0 else 0.0

    return {
        "duration": round(dur, 3),
        "f0_med": round(f0_med, 2),
        "f0_sd_st": round(f0_sd, 3),
        "f0_range_st": round(max(semis) - min(semis), 3) if semis else 0.0,
        "rms_db": round(dsp.rms_db(samples), 2),
        "pause_ratio": round(pause_ratio, 3),
        "zcr_quiet": round(zcr_q, 4),
        "quiet_rms_db": round(quiet_rel_db, 2),
        "hf_ratio": round(hf_ratio, 4),
        "interval_entropy": round(interval_entropy(f0s), 3),
        "rate_syl": round(rate_syl, 2),
        "voiced_ratio": round(len(voiced) / float(len(f0s)), 3) if f0s else 0.0,
    }


def fingerprint_wav(wav_bytes: bytes) -> dict:
    x, rate = dsp.parse_wav(wav_bytes)
    return fingerprint(x, rate)


# ───────────────────────── 方向性断言 ─────────────────────────

# 每种情绪相对「平静」期望的方向：字段 → +1 变大 / -1 变小
EXPECT: dict[str, dict[str, int]] = {
    "兴奋": {"f0_med": 1, "rms_db": 1, "rate_syl": 1, "f0_sd_st": 1},
    "开心": {"f0_med": 1, "rms_db": 1, "f0_sd_st": 1},
    "生气": {"rms_db": 1, "rate_syl": 1},
    "撒娇": {"f0_med": 1, "hf_ratio": 1},
    "温柔": {"rms_db": -1, "hf_ratio": 1},
    "难过": {"f0_med": -1, "rms_db": -1, "pause_ratio": 1},
    "疲惫": {"f0_med": -1, "rms_db": -1, "rate_syl": -1},
    "哭腔": {"f0_sd_st": 1, "pause_ratio": 1, "hf_ratio": 1},
    "害羞": {"hf_ratio": 1, "f0_med": 1},
    "感动": {"f0_sd_st": 1, "hf_ratio": 1},
    "调皮": {"f0_sd_st": 1, "rate_syl": 1},
    "急切": {"rate_syl": 1, "rms_db": 1},
    "冷淡": {"f0_sd_st": -1, "rms_db": -1},
}


def check_direction(base: dict, target: dict, label: str,
                    eps: dict | None = None) -> list[dict]:
    """对比两份指纹，返回逐项判定（passed 字段）。"""
    eps = eps or {"f0_med": 2.0, "rms_db": 0.7, "rate_syl": 0.15,
                  "f0_sd_st": 0.15, "pause_ratio": 0.02, "quiet_rms_db": 0.5,
                  "hf_ratio": 0.03}
    rules = EXPECT.get(label, {})
    rows = []
    for key, sign in rules.items():
        if key not in base or key not in target:
            continue
        delta = target[key] - base[key]
        need = eps.get(key, 0.0)
        ok = (delta * sign) > need
        rows.append({
            "metric": key, "base": base[key], "target": target[key],
            "delta": round(delta, 3), "want": "+" if sign > 0 else "-",
            "passed": bool(ok),
        })
    return rows


def compare(a: dict, b: dict) -> dict:
    keys = set(a) & set(b)
    return {k: round(b[k] - a[k], 3) for k in sorted(keys) if isinstance(a[k], (int, float))}