# GPT-SoVITS 情感层（sovits-emotion-kit）

> **你只需要一个能跑的 GPT-SoVITS。** 不用训练、不用微调、不用换模型。
> 给它一段你自己的干净参考音频（3~8 秒），它就能按情绪换参考音频（ref）来念，
> 情绪是**模型自己演出来的**，不是合成完再加变调/气声那种后期。

- 纯 Python 标准库，零第三方依赖
- 不改你的 GPT-SoVITS，也不碰模型权重
- 提供两种接法：独立 `/tts` 服务（最省事）、或接到你已有的 TTS 网关里
- 附一个自检脚本，一条命令就能验证"情绪真的出来了"

---

## 1. 为什么"换参考音频"才是对的做法

GPT-SoVITS 的情绪来自 **in-context learning**：你给它一段"什么情绪、什么音色"的参考音频 +
逐字转写，它就用那种韵律念你的文本。

| 做法 | 情绪从哪来 | 结果 |
|---|---|---|
| ❌ 先中性念，再对波形变调/加气声 | 后处理 | 音色发飘、韵律被抹平，还可能出电音/回音 |
| ✅ 按情绪换 ref + 采样参数 | **模型内部** | 音高、气声、语速、重音一起变，听着是"她在演" |

所以本项目的核心就是一张表：**情绪 → (参考音频, prompt_text, temperature, top_k, top_p,
repetition_penalty, fragment_interval, speed_factor)**，见 `emotion/refs.py`。

---

## 2. 五分钟接入

### 2.0 前置

- Python 3.9+
- 一个在跑的 GPT-SoVITS（`api_v2.py`，默认 `http://127.0.0.1:9880`）
- 一段**你自己音色**的干净录音：单人、无背景音、中性语气、3~8 秒（最长 12 秒）

```bash
git clone https://github.com/<you>/sovits-emotion-kit.git
cd sovits-emotion-kit
cp config.example.json config.json      # Windows: copy config.example.json config.json
```

编辑 `config.json`，至少填这三项：

```json
"gsv_url": "http://127.0.0.1:9880",
"base_ref": "D:/voice/my-ref.wav",
"prompt_text": "这段录音的逐字转写",
"prompt_lang": "all_zh"
```

> `prompt_text` **必须和音频内容逐字一致**。差一个字，文字和韵律就会错位，念出来会飘。

### 2.1 生成 10 个情绪 ref

```bash
python tools/derive_refs.py --config config.json
```

它从你那段基准 ref 派生出 平静/开心/撒娇/温柔/难过/生气/疲惫/冷淡/惊讶/害怕 十个 ref
（用"重采样 + WSOLA 拉回时长"做变调，时长不变、不带相位伪影），并写出 `refs/emotional/index.json`。

```bash
python tools/derive_refs.py --config config.json --list     # 看每个 ref 的指纹
```

**更好的做法**：每个情绪录一段真人素材，直接覆盖 `refs/emotional/<情绪>.wav`，
再改 `index.json` 里那一项的 `prompt_text` / `prompt_lang`（`--force` 会覆盖你的真人 ref，别乱用）。

### 2.2 自检

```bash
python tools/doctor.py --config config.json
```

它会检查：ref 是否齐全、音高有没有梯度、SoVITS 是否在线，然后真的合成"平静 vs 难过"两段，
比较客观指标（F0 / 语速 / 能量），把试听丢到 `doctor_ab/`。

### 2.3 起服务

```bash
python server/emotion_server.py --config config.json --port 9883
```

```bash
curl -X POST http://127.0.0.1:9883/tts ^
  -H "Content-Type: application/json" ^
  -d "{\"text\":\"今天有点累，不过看到你就好啦\"}" ^
  -o out.wav
```

接口：

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | ref 数量、SoVITS 连通性、开关状态 |
| GET | `/refs` | 每个情绪 ref 的指纹（F0/能量/气声） |
| POST | `/tts` | `{"text","emotion"?,"text_lang"?,"lang"?}` → `audio/wav`（响应头 `X-Emotion`、`X-Ref`） |
| POST | `/analyze` | 只判情绪，返回 VAD + 逐小句弧线 |

`emotion` 字段可以强制情绪（`"难过"`、`"撒娇"`…），不给就按文本自动判。

---

## 3. 接到已有网关里（不想换服务）

看 [`bridge/SERVICE_PATCH.md`](bridge/SERVICE_PATCH.md)：把一个 `emotion_bridge.py` 放进你的服务，
在调 SoVITS 之前 `body.update(emotion_for(...))` 就行，三处改动。

---

## 4. 情绪是怎么判的

```
文本 → analyzer：情感词典 + VAD(效价/唤醒/支配) + 逐小句弧线 + 情绪惯性
     → planner ：情绪强度连续插值 → 音高曲线 / 语速 / 颤音 / 气声 / 停顿 / 句尾
     → refs    ：情绪 → ref + 采样参数（这就是"模型侧情感"）
     → 你的 SoVITS → 拼接 → 去尾巴
```

17 个细标签映射到 10 个 ref 原型（兴奋→开心，委屈/哭腔→难过，害羞/调皮→撒娇…），
所以"70% 的难过"这种连续强度也能表达。

---

## 5. 参数怎么调

| 想改什么 | 改哪里 |
|---|---|
| 情绪的强弱 | `emotion/refs.py` → `DERIVE`（semitones / gain_db） |
| 采样风格（活一点/稳一点） | `emotion/refs.py` → `SAMPLING`（temperature / rep / fragment_interval） |
| 角色的语气底色 | `personas/*.json` → `baseline` / `emotion_bias` |
| 长句里转折要不要换音色 | `config.json` → `per_segment_ref`（true=按小句换 ref，转折明显但音色会跳） |
| 音质最优先 | `config.json` → `clean_refs: true`（完全不动 ref，只调参数） |
| 句尾混响感 | `config.json` → `tail_trim_db`（默认 -45，越小剪越狠） |

---

## 6. 实测数据（供参考，你的音色会不同）

同一句话、同一个 seed、只换 ref：

| 情绪 | ΔF0 | Δ语速 | Δ能量 |
|---|---|---|---|
| 开心 | **+27Hz** | −0.8 字/秒 | +0.5dB |
| 撒娇 | **+51Hz** | −0.1 | −0.6dB |
| 温柔 | **−23Hz** | **−2.4** | −0.9dB |
| 难过 | **−38Hz** | **−1.9** | −1.2dB |
| 生气 | +16Hz | −1.8 | **+1.2dB** |

长句（4 小句）会自动走弧线：平静 → 开心 → **难过**（更慢更沉）→ 平静。

---

## 7. 常见坑（都踩过）

| 现象 | 原因 | 解 |
|---|---|---|
| 电音 / 金属味 | 拿"加气声 / OLA 变速"处理过的音频当 ref，模型把它当音色学走 | 只用变调（本项目用重采样+WSOLA），不加速、不加噪 |
| 回音 / 字被加重音 | WSOLA 搜索位置累积漂移，某段被放两遍 | 名义位置自己推进，搜索只决定取哪一帧（已修） |
| 念出来飘、字乱 | `prompt_text` 和 ref 音频内容不一致 | 逐字对齐；或用 `all_xx` 明确语言 |
| 长句情绪突变全丢 | 每个小句被整句标签覆盖 | 用小句自己的标签（VAD 已混过，不会抖） |
| 一句话音色变了 | 逐小句换 ref | 默认按整句主情绪选 ref；要转折才开 `per_segment_ref` |
| 断句怪 | 文本里有 emoji / 表情包标记 | 送 TTS 前先剥掉（`emotion.lexicon.clean_for_speech`） |

---

## 8. 目录

```
emotion/            情感引擎（词典/分析/韵律/ref/渲染/DSP/指标，纯标准库）
server/emotion_server.py   独立 HTTP 服务（/tts /analyze /health /refs）
tools/derive_refs.py       生成 10 个情绪 ref
tools/doctor.py            自检 + 端到端 A/B
bridge/                    接进已有网关的补丁与改动记录
personas/                  角色语气模板（JSON）
refs/                      情绪 ref 与 index.json（.gitignore 里忽略音频）
```

`python -m emotion.selftest` 可以跑 39 项离线自检（毫秒级，不需要模型）。

## 9. 许可

MIT。模型权重、音色素材、参考音频都不在本仓库里 —— 那些自己准备，注意授权与肖像权。