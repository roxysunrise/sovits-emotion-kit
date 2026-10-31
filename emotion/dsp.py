"""dsp —— 纯标准库音频处理（不装 numpy 也能跑）。

为什么要自己写：
    worker.py 跑在 conda base（Python 3.13），那台机器上没有 numpy/scipy。
    情感要落地成「音高曲线 / 气声 / 颤栗 / 停顿」，必须自己能动样本。
    所以这里只有 struct/math/random，全部 PCM16 单声道进出。

五个动作对应五种情感手段（planner 给参数，这里执行）：
    pitch_contour   颗粒重采样变调 + 颤音 + 颤栗 → 音高曲线
    add_breath      安静段注入带通噪声        → 气声/呼吸感
    stretch_tail    句尾 WSOLA 拉长            → 拖音、舍不得
    silence         句间停顿                   → 犹豫、哽咽
    normalize_rms   响度归一到 -20dBFS + 软限幅 → 不爆音、不忽大忽小
"""
from __future__ import annotations

import math
import random
import struct

PCM16_MAX = 32767.0


# ───────────────────────── WAV 读写 ─────────────────────────

def parse_wav(data: bytes) -> tuple[list[float], int]:
    """解析 PCM WAV → (样本 float [-1,1], 采样率)。多声道自动混单声道。"""
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError("不是 WAV")
    rate = 24000
    ch = 1
    bits = 16
    pcm = b""
    off = 12
    while off + 8 <= len(data):
        tag = data[off:off + 4]
        size = struct.unpack("<I", data[off + 4:off + 8])[0]
        body = data[off + 8: off + 8 + size]
        if tag == b"fmt " and len(body) >= 16:
            codec, ch, rate, _br, _al, bits = struct.unpack("<HHIIHH", body[:16])
            if codec != 1:
                raise ValueError(f"非 PCM（codec={codec}），先用 ffmpeg 转")
        elif tag == b"data":
            pcm = body
        off += 8 + size + (size % 2)
    if not pcm:
        raise ValueError("没有 data 段")
    n_ch = max(1, ch)
    if bits == 16:
        fmt = f"<{len(pcm)//2}h"
        raw = struct.unpack(fmt, pcm[: (len(pcm) // 2) * 2])
        div = PCM16_MAX
    elif bits == 8:
        raw = struct.unpack(f"<{len(pcm)}B", pcm)
        div = 128.0
        raw = [v - 128 for v in raw]
    elif bits == 24:
        raw = []
        for i in range(0, len(pcm) - 2, 3):
            v = pcm[i] | (pcm[i + 1] << 8) | (pcm[i + 2] << 16)
            if v & 0x800000:
                v -= 1 << 24
            raw.append(v)
        div = 8388608.0
    elif bits == 32:
        raw = struct.unpack(f"<{len(pcm)//4}i", pcm[: (len(pcm) // 4) * 4])
        div = 2147483648.0
    else:
        raise ValueError(f"不支持的位深 {bits}")

    if n_ch == 1:
        out = [v / div for v in raw]
    else:
        out = []
        frames = len(raw) // n_ch
        for f in range(frames):
            base = f * n_ch
            out.append(sum(raw[base:base + n_ch]) / (n_ch * div))
    return out, rate


def build_wav(samples: list[float], rate: int = 24000) -> bytes:
    """float 样本 → PCM16 单声道 WAV 字节。"""
    n = len(samples)
    data = bytearray(n * 2)
    for i, s in enumerate(samples):
        v = int(max(-1.0, min(1.0, s)) * PCM16_MAX)
        struct.pack_into("<h", data, i * 2, v)
    hdr = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVE"
    hdr += b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
    hdr += b"data" + struct.pack("<I", len(data))
    return hdr + bytes(data)


# ───────────────────────── 基础工具 ─────────────────────────

def duration(samples: list[float], rate: int) -> float:
    return len(samples) / float(rate or 1)


def silence(rate: int, seconds: float) -> list[float]:
    return [0.0] * max(1, int(rate * max(0.0, seconds)))


def _hann(n: int) -> list[float]:
    if n <= 1:
        return [1.0] * max(1, n)
    return [0.5 - 0.5 * math.cos(2 * math.pi * i / (n - 1)) for i in range(n)]


def _contour_at(contour, x: float) -> float:
    if not contour:
        return 0.0
    if x <= contour[0][0]:
        return contour[0][1]
    for i in range(len(contour) - 1):
        x0, y0 = contour[i]
        x1, y1 = contour[i + 1]
        if x0 <= x <= x1:
            k = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
            return y0 + (y1 - y0) * k
    return contour[-1][1]


def frame_rms(x: list[float], rate: int, frame_ms: float = 10.0) -> list[float]:
    n = len(x)
    fl = max(1, int(rate * frame_ms / 1000.0))
    out = []
    for i in range(0, n, fl):
        seg = x[i:i + fl]
        if not seg:
            out.append(0.0)
            continue
        s = 0.0
        for v in seg:
            s += v * v
        out.append(math.sqrt(s / len(seg)))
    return out


def _expand(frames: list[float], n: int, frame_len: int) -> list[float]:
    """帧级值 → 样本级（线性插值）。"""
    out = [0.0] * n
    if not frames:
        return out
    for i in range(n):
        fpos = i / float(frame_len or 1)
        i0 = int(fpos)
        if i0 >= len(frames) - 1:
            out[i] = frames[-1]
        else:
            k = fpos - i0
            out[i] = frames[i0] * (1 - k) + frames[i0 + 1] * k
    return out


# ───────────────────────── 变调 / 颤音 / 颤栗 ─────────────────────────

def pitch_contour(
    x: list[float],
    rate: int,
    contour: list[tuple[float, float]] | None = None,
    base_semitones: float = 0.0,
    vibrato_hz: float = 0.0,
    vibrato_st: float = 0.0,
    tremor: float = 0.0,
    frame_ms: float = 40.0,
    hop_ms: float = 20.0,
    seed: int = 7,
) -> list[float]:
    """颗粒式变调：帧内重采样 + 汉宁窗 OLA，时长不变。

    contour 是 [(0~1 位置, 半音)] 的分段线性曲线 → 句内音高走向。
    颤音是周期 LFO，颤栗是非周期随机抖动（哭腔的「抖」不是抖音）。
    """
    n = len(x)
    if n < 32:
        return list(x)
    flat = (not contour or all(abs(y) < 1e-4 for _, y in contour))
    if flat and abs(base_semitones) < 1e-3 and vibrato_st < 1e-3 and tremor < 1e-3:
        return list(x)

    frame = max(64, int(rate * frame_ms / 1000.0))
    hop = max(16, int(rate * hop_ms / 1000.0))
    win = _hann(frame)
    out = [0.0] * (n + frame)
    wsum = [0.0] * (n + frame)
    rnd = random.Random(seed)

    pos = 0
    while pos < n:
        frac = pos / float(n - 1) if n > 1 else 0.0
        st = base_semitones + _contour_at(contour, frac)
        if vibrato_st:
            st += vibrato_st * math.sin(2 * math.pi * vibrato_hz * (pos / float(rate)))
        if tremor:
            st += tremor * (rnd.random() * 2.0 - 1.0)
        r = 2.0 ** (st / 12.0)
        center = pos + frame * 0.5
        start = center - frame * r * 0.5
        for k in range(frame):
            sp = start + k * r
            i0 = int(sp)
            s = 0.0
            if 0 <= i0 < n:
                s = x[i0]
                if i0 + 1 < n:
                    s += (x[i0 + 1] - x[i0]) * (sp - i0)
            w = win[k]
            out[pos + k] += s * w
            wsum[pos + k] += w
        pos += hop

    res = [0.0] * n
    for i in range(n):
        w = wsum[i]
        res[i] = out[i] / w if w > 1e-6 else out[i]
    return res


# ───────────────────────── 气声 ─────────────────────────

# ───────────────────────── 相位声码器变调（ref 派生用）─────────────────────────

def _fft2(a: list[complex]) -> None:
    """就地的迭代 radix-2 FFT，长度必须是 2 的幂。"""
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
    ln = 2
    while ln <= n:
        ang = -2.0 * math.pi / ln
        wlen = complex(math.cos(ang), math.sin(ang))
        half = ln >> 1
        for i in range(0, n, ln):
            w = complex(1.0, 0.0)
            for k in range(i, i + half):
                u = a[k]
                v = a[k + half] * w
                a[k] = u + v
                a[k + half] = u - v
                w *= wlen
        ln <<= 1


def _ifft2(a: list[complex]) -> None:
    """就地逆 FFT（共轭法），长度同 _fft2。"""
    n = len(a)
    for i in range(n):
        a[i] = a[i].conjugate()
    _fft2(a)
    inv = 1.0 / n
    for i in range(n):
        a[i] = a[i].conjugate() * inv


def _resample_linear(x: list[float], factor: float) -> list[float]:
    """线性插值重采样（改变音高，同时改变时长）。factor>1 = 变长变低沉。"""
    if abs(factor - 1.0) < 1e-4 or not x:
        return list(x)
    n_out = max(2, int(len(x) * factor))
    out = [0.0] * n_out
    last = len(x) - 1
    for i in range(n_out):
        sp = i / factor
        i0 = int(sp)
        if i0 >= last:
            out[i] = x[last]
        else:
            k = sp - i0
            out[i] = x[i0] * (1.0 - k) + x[i0 + 1] * k
    return out


def _wsola(x: list[float], rate: int, factor: float,
           frame_ms: float = 48.0, tol_ms: float = 12.0, step: int = 4) -> list[float]:
    """WSOLA 时长伸缩：不需要相位，所以没有相位声码器的"水声/嗡嗡"、
    也没有颗粒法的帧率调制。

    ⚠️ 2026-10-31 修了一个会造成「回音 + 重音」的真 bug：
    旧版把 `pos_in` 写成 `best + hop_a`（best 已经含搜索偏移），位置会一路累积漂移，
    于是某段音频被放两遍、另一段被跳过 —— 听感就是"回音"加上"字被加重音"。
    正确做法（Verhelst & Roelands）：名义位置 `a_k += hop_a` 自己推进，
    搜索只决定"取哪一帧"，不改变名义位置。

    相似度匹配也对齐了：候选帧的**头部**要和上一帧在输入里的**自然延续段**
    （x[prev+hop_s : prev+2hop_s]）最像，而不是和上一帧的尾部比。
    """
    if abs(factor - 1.0) < 1e-3 or len(x) < 512:
        return list(x)
    frame = max(256, int(rate * frame_ms / 1000.0))
    hop_s = frame // 2
    hop_a = max(1, int(round(hop_s * factor)))
    tol = max(1, int(rate * tol_ms / 1000.0))
    win = _hann(frame)
    n_out = max(frame, int(len(x) * factor))
    out = [0.0] * (n_out + frame)
    wsum = [0.0] * (n_out + frame)

    a_k = 0
    pos_out = 0
    prev_best = None
    while pos_out + frame <= len(out) and a_k + frame <= len(x):
        if prev_best is None:
            best = a_k
        else:
            ref_start = prev_best + hop_s
            lo = max(0, a_k - tol)
            hi = min(len(x) - frame - 1, a_k + tol)
            if hi < lo:
                hi = lo = max(0, min(len(x) - frame - 1, a_k))
            ref_end = ref_start + hop_s
            if ref_end > len(x):
                ref_end = len(x)
            best = lo
            best_score = -1e18
            cand = lo
            while cand <= hi:
                s = 0.0
                i = 0
                while i < hop_s and cand + i < len(x) and ref_start + i < ref_end:
                    s += x[cand + i] * x[ref_start + i]
                    i += 1
                if s > best_score:
                    best_score = s
                    best = cand
                cand += step
            if best_score <= 0:
                best = max(0, min(len(x) - frame - 1, a_k))
        for i in range(frame):
            out[pos_out + i] += x[best + i] * win[i]
            wsum[pos_out + i] += win[i]
        prev_best = best
        a_k += hop_a
        pos_out += hop_s

    for i in range(len(out)):
        if wsum[i] > 1e-6:
            out[i] /= wsum[i]
    return out[:n_out]


def pitch_shift_wsola(x: list[float], rate: int, semitones: float) -> list[float]:
    """重采样 + WSOLA 拉回时长 = 变调但时长不变，且没有相位类伪影。

    比相位声码器的"嗡嗡"和颗粒法的"电子颤音"都干净 —— 给 ref 用这个。
    """
    if abs(semitones) < 0.05 or len(x) < 256:
        return list(x)
    r = 2.0 ** (semitones / 12.0)
    # 变调：resample 到 1/r 长度（音高 ×r）
    y = _resample_linear(x, 1.0 / r)
    # 时长拉回：×r
    z = _wsola(y, rate, r)
    if len(z) < len(x):
        z = z + [0.0] * (len(x) - len(z))
    return z[:len(x)]


def pitch_shift_pv(x: list[float], rate: int, semitones: float,
                   win_ms: float = 64.0, hop_ms: float = 16.0) -> list[float]:
    """相位声码器变调：时长不变，相位连续 —— 不会像颗粒法那样带"金属味/颤音"。

    为什么 ref 必须用这个：颗粒式变调（`pitch_contour`）每帧重采样会打断相位，
    听起来像加了电子颤音；而 GPT-SoVITS 会把 ref 的音质当音色学走，
    于是输出就"电"。实测电音指标（谱平坦度）0.44→0.71 就是这么来的。

    做法：STFT → 每个 bin 的幅度按比例搬到目标 bin（线性插值）→ 相位由
    该 bin 的中心频率逐帧累加（相位锁定）→ IFFT + 窗归一化 OLA。
    """
    if abs(semitones) < 0.05 or len(x) < 256:
        return list(x)
    n = len(x)
    win_len = 1
    target = max(512, int(rate * win_ms / 1000.0))
    while win_len * 2 <= target:
        win_len *= 2
    hop = max(64, int(rate * hop_ms / 1000.0))
    win = _hann(win_len)
    win_sum_sq = [w * w for w in win]

    pad = win_len
    buf = [0.0] * pad + list(x) + [0.0] * (win_len + hop)
    total = len(buf)
    frames = max(1, (total - win_len) // hop + 1)
    out = [0.0] * (total + win_len)
    wsum = [0.0] * (total + win_len)

    ratio = 2.0 ** (semitones / 12.0)
    bins = win_len // 2 + 1
    phase_acc = [0.0] * bins
    # 每个 bin 的中心角频率（每 hop 的相位增量）
    step = [2.0 * math.pi * hop * k / win_len for k in range(bins)]

    for f in range(frames):
        start = f * hop
        seg = buf[start:start + win_len]
        spec = [complex(seg[i] * win[i], 0.0) for i in range(win_len)]
        _fft2(spec)
        mags = [abs(spec[k]) for k in range(bins)]
        new_mag = [0.0] * bins
        for k in range(1, bins - 1):
            m = mags[k]
            if m <= 1e-9:
                continue
            src = k * ratio
            k0 = int(src)
            if k0 >= bins - 1:
                continue
            frac = src - k0
            new_mag[k0] += m * (1.0 - frac)
            new_mag[k0 + 1] += m * frac
        # 相位：按 bin 自身频率推进（相位锁定），避免相邻帧相位乱跳
        for k in range(bins):
            phase_acc[k] += step[k]
        frame = [0j] * win_len
        for k in range(bins):
            frame[k] = complex(new_mag[k] * math.cos(phase_acc[k]),
                               new_mag[k] * math.sin(phase_acc[k]))
        # 共轭对称，保证 IFFT 结果是实数
        for k in range(1, bins - 1):
            frame[win_len - k] = frame[k].conjugate()
        _ifft2(frame)
        for i in range(win_len):
            out[start + i] += frame[i].real * win[i]
            wsum[start + i] += win_sum_sq[i]

    res = [0.0] * len(out)
    for i in range(len(out)):
        res[i] = out[i] / wsum[i] if wsum[i] > 1e-6 else 0.0
    # 去掉补零段
    return res[pad:pad + n]


def trim_tail(x: list[float], rate: int, thresh_db: float = -45.0,
              hold_ms: float = 60.0, floor_ms: float = 250.0) -> list[float]:
    """剪掉句末的弱拖尾（模型的混响尾巴）—— 听感上的"回音"多半是它。

    只切真正很弱的部分（默认 -45dB 以下），并保留 hold_ms 的自然收音，
    所以不会把词尾的拖长音切掉；整体时长低于 floor_ms 就不动。
    注意：这是**去尾巴**，不是加效果，不改变音高/音色/情绪。
    """
    if len(x) < rate * 0.05:
        return list(x)
    fl = max(1, int(rate * 0.01))
    rms = frame_rms(x, rate, 10.0)
    if not rms:
        return list(x)
    peak = max(rms)
    if peak <= 1e-9:
        return list(x)
    thresh = peak * (10.0 ** (thresh_db / 20.0))
    last = -1
    for i, v in enumerate(rms):
        if v >= thresh:
            last = i
    if last < 0:
        return list(x)
    keep = int((last + 1) * fl + rate * hold_ms / 1000.0)
    keep = max(int(rate * floor_ms / 1000.0), min(keep, len(x)))
    return x[:keep]


def add_breath(x: list[float], rate: int, amount: float, seed: int = 11) -> list[float]:
    """在能量低的地方注入带通噪声 → 气声、呼吸感。

    情感上：温柔/撒娇/疲惫/哭腔靠气声；生气/兴奋几乎不加。
    安静处加得多，是为了模拟「话缝里的换气」，而不是把整句糊上一层噪声。
    """
    if amount <= 0.001 or len(x) < 64:
        return list(x)
    n = len(x)
    rnd = random.Random(seed)
    lp1 = lp2 = 0.0
    a_lp, a_hp = 0.62, 0.05
    noise = [0.0] * n
    for i in range(n):
        w = rnd.uniform(-1.0, 1.0)
        lp1 += a_lp * (w - lp1)
        lp2 += a_hp * (w - lp2)
        noise[i] = lp1 - lp2

    fl = max(1, int(rate * 0.01))
    rms = frame_rms(x, rate, 10.0)
    peak = max(rms) if rms else 0.0
    if peak <= 1e-6:
        return list(x)
    env = _expand(rms, n, fl)

    # 气声自身的响度：跟着人声响度走，静音处再抬一点（模拟话缝里的换气）
    base = peak * (0.35 + 0.90 * amount)
    out = [0.0] * n
    for i in range(n):
        e = env[i] / peak
        g = 0.30 + 0.70 * (1.0 - min(1.0, e)) if e < 0.6 else 0.22
        out[i] = x[i] + noise[i] * base * g
    return out


# ───────────────────────── 句尾拖长 ─────────────────────────

def stretch_tail(x: list[float], rate: int, factor: float, last_ms: float = 200.0) -> list[float]:
    """把最后 last_ms 拉长 factor 倍（WSOLA 带相似度搜索，避免咔哒）。"""
    if factor <= 1.001 or len(x) < 256:
        return list(x)
    tail_len = min(int(rate * last_ms / 1000.0), len(x) // 2)
    if tail_len < 128:
        return list(x)
    head = x[: len(x) - tail_len]
    tail = x[len(x) - tail_len:]

    frame = max(64, int(rate * 0.03))
    hop_out = frame // 2
    hop_in = max(1, int(hop_out / factor))
    search = max(1, int(rate * 0.004))
    win = _hann(frame)
    target = int(len(tail) * factor)
    out = [0.0] * (target + frame)
    wsum = [0.0] * (target + frame)

    out_pos = 0
    in_pos = 0
    while out_pos < target:
        # 在 in_pos ± search 里找与上一个帧最相似的位置
        best = in_pos
        if 0 < out_pos:
            best_corr = -1e18
            for d in range(-search, search + 1):
                p = in_pos + d
                if p < 0 or p + frame > len(tail):
                    continue
                corr = 0.0
                for k in range(0, frame, 4):
                    corr += tail[p + k] * out[out_pos + k] if out_pos + k < len(out) else 0.0
                if corr > best_corr:
                    best_corr = corr
                    best = p
            in_pos = best if best >= 0 else in_pos
        for k in range(frame):
            src = in_pos + k
            if 0 <= src < len(tail) and out_pos + k < len(out):
                out[out_pos + k] += tail[src] * win[k]
                wsum[out_pos + k] += win[k]
        out_pos += hop_out
        in_pos += hop_in

    res = []
    for i in range(target):
        w = wsum[i]
        res.append(out[i] / w if w > 1e-6 else out[i])
    return head + res


# ───────────────────────── 增益 / 响度 ─────────────────────────

def apply_gain(x: list[float], db: float) -> list[float]:
    if abs(db) < 0.01:
        return list(x)
    g = 10.0 ** (db / 20.0)
    return [v * g for v in x]


def rms_db(x: list[float]) -> float:
    if not x:
        return -120.0
    s = 0.0
    for v in x:
        s += v * v
    r = math.sqrt(s / len(x))
    return 20.0 * math.log10(max(r, 1e-9))


def normalize_rms(x: list[float], target_db: float = -20.0, ceiling: float = 0.97) -> list[float]:
    """响度归一 + 软限幅。QQ 那头解码后最容易爆音，这一步别省。"""
    if not x:
        return []
    cur = rms_db(x)
    g = 10.0 ** ((target_db - cur) / 20.0)
    out = []
    for v in x:
        y = v * g
        # 软限幅：超过 ceiling 的部分按 tanh 压回去，听感上不「炸」
        lim = ceiling * math.tanh(abs(y) / max(ceiling, 1e-6))
        out.append(lim if abs(y) > ceiling else y)
    return out


def trim_silence(x: list[float], rate: int, thresh_db: float = -42.0,
                 pad_ms: float = 12.0, max_trim_ms: float = 400.0) -> list[float]:
    """去掉首尾静音（edge-tts 常带几百毫秒空白，会在 QQ 里显得呆）。"""
    if not x:
        return []
    th = 10.0 ** (thresh_db / 20.0)
    fl = max(1, int(rate * 0.01))
    # 头
    start = 0
    for i in range(0, len(x), fl):
        if max(abs(v) for v in x[i:i + fl]) > th:
            start = i
            break
    end = len(x)
    for i in range(len(x) - fl, -1, -fl):
        if max(abs(v) for v in x[i:i + fl]) > th:
            end = min(len(x), i + fl)
            break
    pad = int(rate * pad_ms / 1000.0)
    limit = int(rate * max_trim_ms / 1000.0)
    start = max(0, start - pad) if start <= limit else start - pad
    end = min(len(x), end + pad) if (len(x) - end) <= limit else end + pad
    return x[start:end]


def insert_at_quiet(x: list[float], rate: int, seconds: float,
                    lo: float = 0.40, hi: float = 0.78) -> list[float]:
    """在句子最安静的地方塞一小段停顿 —— 犹豫、哽咽、换气都靠它。

    位置不是拍脑袋的 50%：在 [lo, hi] 这一段里找能量最低的 20ms 帧，
    那多半正好是词与词之间的缝隙，插在这儿听不出来是「贴上去的」。
    """
    if seconds <= 0.01 or len(x) < rate // 2:
        return list(x)
    fl = max(1, int(rate * 0.02))
    frames = frame_rms(x, rate, 20.0)
    start = int(len(frames) * lo)
    end = max(start + 1, int(len(frames) * hi))
    idx = min(range(start, end), key=lambda i: frames[i]) if end > start else len(frames) // 2
    pos = idx * fl
    fade = max(8, int(rate * 0.006))
    gap = [0.0] * int(rate * seconds)
    # 两端做 6ms 淡出淡入，避免咔哒
    head = list(x[:pos])
    tail = list(x[pos:])
    for i in range(min(fade, len(head))):
        head[len(head) - 1 - i] *= 1.0 - i / float(fade)
    for i in range(min(fade, len(tail))):
        tail[i] *= i / float(fade)
    return head + gap + tail


def crossfade_concat(parts: list[list[float]], rate: int, fade_ms: float = 8.0) -> list[float]:
    """拼接（短交叉淡化，防止接缝咔一声）。"""
    if not parts:
        return []
    fade = max(1, int(rate * fade_ms / 1000.0))
    out = list(parts[0])
    for seg in parts[1:]:
        if not seg:
            continue
        f = min(fade, len(out), len(seg))
        if f > 1:
            for i in range(f):
                k = i / float(f)
                out[len(out) - f + i] = out[len(out) - f + i] * (1 - k) + seg[i] * k
            out.extend(seg[f:])
        else:
            out.extend(seg)
    return out