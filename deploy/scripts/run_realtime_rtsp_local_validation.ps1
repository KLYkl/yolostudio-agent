param(
    [Parameter(Mandatory = $true)]
    [string]$RtspUrl,

    [Parameter(Mandatory = $true)]
    [string]$Model,

    [ValidateSet("auto", "yolo", "yolodo")]
    [string]$EnvName = "auto",

    [string]$OutputRoot = "",
    [int]$TimeoutMs = 5000,
    [int]$FrameIntervalMs = 120,
    [int]$MaxFrames = 8,
    [double]$WaitSeconds = 20,
    [double]$PollIntervalSeconds = 0.5,
    [string]$LocalCondaRoot = "",
    [switch]$KeepRunning
)

$ErrorActionPreference = "Stop"
$RepoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $RepoRoot ".tmp_realtime_rtsp_validation"
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
    foreach ($candidate in @("yolo", "yolodo")) {
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

$env:SystemRoot = 'C:\Windows'
$env:windir = 'C:\Windows'
$env:COMSPEC = 'C:\Windows\System32\cmd.exe'
$env:PYTHONHASHSEED = "1"

$resolvedEnv = Resolve-LocalPredictEnv -RequestedEnv $EnvName -CondaRoot $LocalCondaRoot
$PythonExe = $resolvedEnv.PythonExe
Write-Host "==> using local conda env: $($resolvedEnv.Name)"

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

$args = @(
    (Join-Path $RepoRoot "deploy\scripts\run_realtime_rtsp_local_validation.py"),
    "--rtsp-url", $RtspUrl,
    "--model", $Model,
    "--output-dir", $OutputRoot,
    "--timeout-ms", "$TimeoutMs",
    "--frame-interval-ms", "$FrameIntervalMs",
    "--max-frames", "$MaxFrames",
    "--wait-seconds", "$WaitSeconds",
    "--poll-interval-seconds", "$PollIntervalSeconds"
)
if ($KeepRunning) {
    $args += "--keep-running"
}

& $PythonExe @args
if ($LASTEXITCODE -ne 0) {
    throw "本地 RTSP 验证失败，exit code=$LASTEXITCODE"
}
