#!/usr/bin/env bash
# 把本目录推到你自己的 GitHub（Linux/macOS）
#   GITHUB_TOKEN=ghp_xxx ./publish.sh sovits-emotion-kit [owner] [--private]
set -euo pipefail
cd "$(dirname "$0")"

REPO="${1:-sovits-emotion-kit}"
OWNER="${2:-}"
PRIVATE="false"
[[ "${3:-}" == "--private" ]] && PRIVATE="true"
: "${GITHUB_TOKEN:?缺 token：export GITHUB_TOKEN=ghp_xxx}"

api() { curl -sS -H "Authorization: Bearer $GITHUB_TOKEN" -H "Accept: application/vnd.github+json" -H "User-Agent: publish-sh" "$@"; }

if [ -z "$OWNER" ]; then
  OWNER=$(api https://api.github.com/user | python3 -c 'import json,sys;print(json.load(sys.stdin)["login"])')
fi
echo "[i] 账号: $OWNER  仓库: $REPO"

if api "https://api.github.com/repos/$OWNER/$REPO" | grep -q '"full_name"'; then
  echo "[i] 仓库已存在，直接推"
else
  api -X POST https://api.github.com/user/repos \
      -d "{\"name\":\"$REPO\",\"private\":$PRIVATE,\"description\":\"GPT-SoVITS emotion layer: per-emotion reference audio, zero deps\"}" > /dev/null
  echo "[i] 已创建 https://github.com/$OWNER/$REPO"
fi

git branch -M main
git remote remove origin 2>/dev/null || true
git remote add origin "https://github.com/$OWNER/$REPO.git"
git push "https://x-access-token:$GITHUB_TOKEN@github.com/$OWNER/$REPO.git" -u main
echo "完成 → https://github.com/$OWNER/$REPO"