# 接进已有的 TTS 路由（真实改造记录）

下面是我们自己那套网关（`voice-router`，FastAPI）的改动，纯示范：任何"收文本 → 调 SoVITS → 返回 wav"的服务都能照抄。

## 1. 放两个文件

- 把 `bridge/emotion_bridge.py` 放到你的服务目录
- 把整个 `emotion/` 包放在某处（比如 `D:/path/to/sovits-emotion-kit`）

## 2. 改三处

**① 导入**

```python
import emotion_bridge
```

**② 合成前覆盖参数**（原来这里是 `body = dict(settings['gsv_body'])` 直接用固定 ref）

```python
body = dict(settings['gsv_body'])
body.update(text=segment.text, text_lang=..., prompt_lang=..., streaming_mode=False)

emo = emotion_bridge.emotion_for(segment.text, settings.get('emotion') or {})
if emo:
    base_speed = float(body.get('speed_factor', 1.0))
    emo_speed = float(emo.pop('speed_factor', 1.0))
    emo.pop('_emotion', ''); emo.pop('_intensity', None)
    body.update(emo)                                  # ref / prompt / temperature / top_k / ...
    body['speed_factor'] = round(max(0.6, min(1.6, base_speed * emo_speed)), 3)
```

**③ 配置里开一个开关**

```json
"emotion": {
  "enabled": true,
  "voice_dir": "D:/path/to/sovits-emotion-kit",
  "refs_dir": "refs/emotional",
  "clean_refs": false,
  "min_confidence": 0.0
}
```

## 3. 注意事项

- **HTTP 头只能 latin-1**：想回情绪标签做 percent 编码（`urllib.parse.quote`），否则中文标签会让响应直接 500。
  我们踩过：`UnicodeEncodeError: 'latin-1' codec can't encode characters`。
- **`/health` 加一个字段**，把 `emotion_bridge.status()` 透出来，线上排查靠它：
  `ok:false` + `err` 空 = 还没被调用过（懒加载）；`ok:false` + `err` 有值 = 加载失败（会退回固定 ref）。
- **别在 RVC/变声之后再换 ref**：情绪必须在最上游那次 SoVITS 推理里就定下来。
  我们是先按情绪合成，再（可选）过 RVC 换音色，情绪跟着走。
- **模型侧提速**：换 ref 会让 SoVITS 重新编码参考音频，单句大约多 100~300ms。
  想省就 `clean_refs: true`（不换 ref），只靠 `speed_factor`/`temperature`。

## 4. 实测（改前 / 改后，同机器同模型）

| 文案 | 改前（固定 ref） | 改后（按情绪换 ref） |
|---|---|---|
| やっと会えたから、すごく嬉しい。 | 287Hz / 2.36 字每秒 / 2.96s | **314Hz** / 3.06 / 2.61s |
| でも、帰り道はちょっと寂しかったな。 | 280Hz / 3.02 字每秒 / 2.98s | **256Hz** / 2.07 / **5.80s** |

两句在改前几乎一样高（287 vs 280）—— 这就是"没情感"的客观样子。