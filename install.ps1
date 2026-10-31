Write-Host '=== GPT-SoVITS 情感层 ===' -ForegroundColor Cyan
Set-Location $PSScriptRoot
python -m emotion.selftest
if (-not (Test-Path 'config.json')) {
    Copy-Item 'config.example.json' 'config.json'
    Write-Host '[i] 已生成 config.json' -ForegroundColor Yellow
    Write-Host '    填好 base_ref / prompt_text / prompt_lang，然后：'
    Write-Host '    python tools/derive_refs.py --config config.json'
    Write-Host '    python tools/doctor.py --config config.json'
} else {
    Write-Host '[i] config.json 已存在，未覆盖'
}
