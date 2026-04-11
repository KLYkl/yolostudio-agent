param(
    [Parameter(Mandatory = $true)]
    [string]$Server,

    [string]$StageDir = "D:\yolodo2.0\agent_plan\.tmp_prediction_real_media_stage",
    [string]$RemoteRoot = "~/prediction_real_media_stage"
)

$ErrorActionPreference = "Stop"

if (!(Test-Path -LiteralPath $StageDir)) {
    throw "StageDir 不存在: $StageDir"
}

$Manifest = Join-Path $StageDir "manifest.json"
if (!(Test-Path -LiteralPath $Manifest)) {
    throw "manifest.json 不存在，请先运行 stage_prediction_real_media.py"
}

Write-Host "==> ensure remote root"
ssh $Server "mkdir -p $RemoteRoot/weights $RemoteRoot/videos"

Write-Host "==> upload manifest"
scp $Manifest "$Server`:$RemoteRoot/manifest.json"

Write-Host "==> upload weights"
Get-ChildItem -LiteralPath (Join-Path $StageDir "weights") -File | ForEach-Object {
    scp $_.FullName "$Server`:$RemoteRoot/weights/$($_.Name)"
}

Write-Host "==> upload videos"
Get-ChildItem -LiteralPath (Join-Path $StageDir "videos") -File | ForEach-Object {
    scp $_.FullName "$Server`:$RemoteRoot/videos/$($_.Name)"
}

Write-Host "upload finished: $RemoteRoot"
