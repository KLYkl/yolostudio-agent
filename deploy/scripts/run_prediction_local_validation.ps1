param(
    [ValidateSet("yolo", "yolodo")]
    [string]$EnvName = "yolo",

    [string]$StageRoot = "D:\yolodo2.0\agent_plan\.tmp_prediction_real_media_stage",
    [string]$OutputRoot = "D:\yolodo2.0\agent_plan\.tmp_prediction_real_media_output"
)

$ErrorActionPreference = "Stop"

if (!(Test-Path -LiteralPath $StageRoot)) {
    throw "StageRoot 不存在: $StageRoot"
}

$WeightsDir = Join-Path $StageRoot "weights"
$VideosDir = Join-Path $StageRoot "videos"
if (!(Test-Path -LiteralPath $WeightsDir)) {
    throw "weights 目录不存在: $WeightsDir"
}
if (!(Test-Path -LiteralPath $VideosDir)) {
    throw "videos 目录不存在: $VideosDir"
}

$PythonExe = "D:\Anaconda\envs\$EnvName\python.exe"
if (!(Test-Path -LiteralPath $PythonExe)) {
    throw "未找到本地 conda 环境 python: $PythonExe"
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

& $PythonExe "D:\yolodo2.0\agent_plan\agent\tests\test_prediction_remote_real_media.py" `
  --weights-dir $WeightsDir `
  --videos-dir $VideosDir `
  --output-dir $OutputRoot
