param(
    [ValidateSet("auto", "yolo", "yolodo")]
    [string]$EnvName = "auto",

    [string]$StageRoot = "",
    [string]$OutputRoot = "",
    [string]$LocalCondaRoot = ""
)

$ErrorActionPreference = "Stop"
$RepoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
if ([string]::IsNullOrWhiteSpace($StageRoot)) {
    $StageRoot = Join-Path $RepoRoot ".tmp_prediction_real_media_stage"
}
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $RepoRoot ".tmp_prediction_real_media_output"
}
if ([string]::IsNullOrWhiteSpace($LocalCondaRoot)) {
    if ($env:YOLO_LOCAL_CONDA_ROOT) {
        $LocalCondaRoot = $env:YOLO_LOCAL_CONDA_ROOT
    }
    else {
        $LocalCondaRoot = Join-Path $env:USERPROFILE "Miniconda3\envs"
    }
}

function Resolve-LocalPredictEnv {
    param(
        [string]$RequestedEnv,
        [string]$CondaRoot
    )

    $candidates = New-Object System.Collections.Generic.List[string]
    if ($RequestedEnv -and $RequestedEnv -ne "auto") {
        $candidates.Add($RequestedEnv)
    }
    foreach ($candidate in @("yolodo", "yolo")) {
        if (-not $candidates.Contains($candidate)) {
            $candidates.Add($candidate)
        }
    }

    foreach ($candidate in $candidates) {
        $pythonExe = Join-Path $CondaRoot "$candidate\python.exe"
        if (Test-Path -LiteralPath $pythonExe) {
            return @{ Name = $candidate; PythonExe = $pythonExe }
        }
    }

    throw "未找到可用本地 conda 环境。已尝试: $($candidates -join ', ')"
}

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

$resolvedEnv = Resolve-LocalPredictEnv -RequestedEnv $EnvName -CondaRoot $LocalCondaRoot
$PythonExe = $resolvedEnv.PythonExe
Write-Host "==> using local conda env: $($resolvedEnv.Name)"

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

$env:PYTHONHASHSEED = "1"
if (-not $env:YOLO_CONFIG_DIR) {
    $env:YOLO_CONFIG_DIR = Join-Path $RepoRoot ".tmp_prediction_local_config"
}

& $PythonExe (Join-Path $RepoRoot "deploy\scripts\run_prediction_local_validation.py") `
  --weights-dir $WeightsDir `
  --videos-dir $VideosDir `
  --output-dir $OutputRoot
