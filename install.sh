#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -m emotion.selftest || true
if [ ! -f config.json ]; then
  cp config.example.json config.json
  echo "[i] 已生成 config.json，填 base_ref / prompt_text / prompt_lang 后："
  echo "    python3 tools/derive_refs.py --config config.json"
  echo "    python3 tools/doctor.py --config config.json"
fi