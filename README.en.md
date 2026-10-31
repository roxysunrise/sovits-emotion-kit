# GPT-SoVITS Emotion Layer (sovits-emotion-kit)

> **You only need a working GPT-SoVITS.** No training, no fine-tuning, no model surgery.
> Give it one clean reference clip (3–8 s) of the target voice and it will *perform* emotion
> by switching the reference audio per emotion — no post-processing pitch/gas tricks.

- Pure Python standard library, zero third-party deps
- Does not modify your GPT-SoVITS or its weights
- Two integration modes: standalone `/tts` server, or a drop-in bridge for your existing gateway
- Ships a doctor script that proves the emotion is actually there

## Why reference audio is the right knob

GPT-SoVITS emotions come from **in-context learning**: the reference clip + its transcript
tell the model *how* to say things. So the mapping this project ships is:

```
emotion -> (ref wav, prompt_text, temperature, top_k, top_p, repetition_penalty,
            fragment_interval, speed_factor)
```

Post-processing the waveform (pitch shift, breath noise) instead smears timbre and can produce
the classic "electric / echo" artifacts. Don't.

## Quick start

```bash
git clone https://github.com/<you>/sovits-emotion-kit.git
cd sovits-emotion-kit
cp config.example.json config.json     # edit gsv_url / base_ref / prompt_text / prompt_lang

python tools/derive_refs.py --config config.json   # builds 10 emotion refs
python tools/doctor.py   --config config.json      # ref check + real A/B synthesis
python server/emotion_server.py --config config.json --port 9883
```

```bash
curl -X POST http://127.0.0.1:9883/tts -H "Content-Type: application/json" \
     -d '{"text":"I am tired today, but seeing you helps."}' -o out.wav
```

`prompt_text` must match the reference audio **word for word**, otherwise prosody drifts.

## Endpoints

| Method | Path | Notes |
|---|---|---|
| GET | `/health` | ref count, upstream reachability, switches |
| GET | `/refs` | fingerprint (F0 / RMS / breath ratio) per emotion ref |
| POST | `/tts` | `{"text","emotion"?,"text_lang"?,"lang"?}` -> `audio/wav` |
| POST | `/analyze` | emotion + per-clause arc only |

## Integrate into an existing gateway

See [`bridge/SERVICE_PATCH.md`](bridge/SERVICE_PATCH.md). Roughly: drop `bridge/emotion_bridge.py`
next to your service and merge the returned dict into your SoVITS request body before calling it.

## Benchmarks (yours will differ)

Same sentence, same seed, only the ref changes:

| emotion | ΔF0 | Δspeech rate | Δlevel |
|---|---|---|---|
| happy | **+27 Hz** | −0.8 syl/s | +0.5 dB |
| cute | **+51 Hz** | −0.1 | −0.6 dB |
| gentle | **−23 Hz** | **−2.4** | −0.9 dB |
| sad | **−38 Hz** | **−1.9** | −1.2 dB |
| angry | +16 Hz | −1.8 | **+1.2 dB** |

Long sentences get an arc automatically (calm -> happy -> sad -> calm).

## Switches

`clean_refs: true` never touches the ref (cleanest timbre, smaller emotional range).
`per_segment_ref: true` switches ref per clause (bigger arc, slight timbre jump).
`tail_trim_db` trims weak reverb tails.

## License

MIT. Model weights, voice data and the reference audio are **not** included — bring your own,
and mind the rights of the voice you clone.