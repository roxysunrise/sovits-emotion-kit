# 把本目录推到你自己的 GitHub（一条命令）
#
#   1) 建一个带 repo 权限的 token：https://github.com/settings/tokens （classic，勾 repo）
#   2) 在 powershell 里：
#        $env:GITHUB_TOKEN = "ghp_xxx"
#        .\publish.ps1 -Repo sovits-emotion-kit
#   3) 可选：-Owner 别的账号/组织   -Private 建私有仓库
#
# 脚本会：token 有效？→ 仓库存在？→ 不存在就用 API 创建一个 → add remote → push main
# 不写任何凭据到磁盘（远端 URL 里不带 token）。
param(
    [string]$Repo = "sovits-emotion-kit",
    [string]$Owner = "",
    [string]$Token = $env:GITHUB_TOKEN,
    [switch]$Private,
    [string]$Description = "GPT-SoVITS emotion layer: per-emotion reference audio, zero deps, no training"
)
$ErrorActionPreference = 'Stop'
if (-not $Token) { throw '缺 token：先 $env:GITHUB_TOKEN = "ghp_xxx"' }

$headers = @{ Authorization = "Bearer $Token"; Accept = "application/vnd.github+json"; "User-Agent" = "publish-ps1" }
if (-not $Owner) {
    $me = Invoke-RestMethod -Uri 'https://api.github.com/user' -Headers $headers -Method Get
    $Owner = $me.login
}
Write-Host "[i] 账号: $Owner  仓库: $Repo" -ForegroundColor Cyan

$exists = $true
try { Invoke-RestMethod -Uri "https://api.github.com/repos/$Owner/$Repo" -Headers $headers -Method Get | Out-Null }
catch { $exists = $false }

if (-not $exists) {
    $body = @{ name = $Repo; description = $Description; private = [bool]$Private; auto_init = $false } | ConvertTo-Json
    Invoke-RestMethod -Uri 'https://api.github.com/user/repos' -Headers $headers -Method Post -Body $body -ContentType 'application/json' | Out-Null
    Write-Host "[i] 已创建仓库 https://github.com/$Owner/$Repo" -ForegroundColor Green
} else {
    Write-Host "[i] 仓库已存在，直接推" -ForegroundColor Yellow
}

git branch -M main
git remote remove origin 2>$null
git remote add origin "https://github.com/$Owner/$Repo.git"
# token 只放在这次 push 的 URL 里，不落盘
git -c http.extraHeader="AUTHORIZATION: basic $([Convert]::ToBase64String([Text.Encoding]::ASCII.GetBytes("x-access-token:$Token")))" push -u origin main

Write-Host ""
Write-Host "完成 → https://github.com/$Owner/$Repo" -ForegroundColor Green
Write-Host "建议随后在仓库 About 里填 topics: gpt-sovits, tts, emotion, voice-cloning"