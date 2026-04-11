param(
    [string]$Server = "yolostudio",
    [ValidateSet("yolo", "yolodo")]
    [string]$EnvName = "yolo",
    [string]$RemoteAppRoot = "/home/kly/yolostudio_agent_proto",
    [string]$RemoteStageRoot = "/home/kly/prediction_real_media_stage",
    [string]$RemoteOutputRoot = "/home/kly/prediction_real_media_output/codex_roundtrip",
    [string]$LocalStageRoot = "D:\yolodo2.0\agent_plan\.tmp_prediction_real_media_stage",
    [string]$LocalResultPath = "D:\yolodo2.0\agent_plan\agent\tests\test_prediction_remote_real_media_output.json"
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true

function Invoke-NativeChecked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Exe,

        [Parameter(Mandatory = $true)]
        [string[]]$Args
    )

    Write-Host ("> " + $Exe + " " + ($Args -join " "))
    & $Exe @Args
    if ($LASTEXITCODE -ne 0) {
        throw "$Exe 执行失败，exit code=$LASTEXITCODE"
    }
}

$repoRoot = "D:\yolodo2.0"
$stageScript = Join-Path $repoRoot "agent_plan\deploy\scripts\stage_prediction_real_media.py"
if (!(Test-Path -LiteralPath (Join-Path $LocalStageRoot "manifest.json"))) {
    Write-Host "==> stage real media locally"
    python $stageScript
    if ($LASTEXITCODE -ne 0) {
        throw "stage_prediction_real_media.py 执行失败"
    }
}

Write-Host "==> ensure remote directories"
Invoke-NativeChecked -Exe "ssh" -Args @(
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=10",
    $Server,
    "mkdir -p $RemoteAppRoot/agent_plan/agent/server/services $RemoteAppRoot/agent_plan/agent/server/tools $RemoteAppRoot/agent_plan/agent/tests $RemoteStageRoot/weights $RemoteStageRoot/videos $RemoteOutputRoot"
)

$syncItems = @(
    @{
        Local = "D:\yolodo2.0\agent_plan\agent\server\services\predict_service.py"
        Remote = "$Server`:$RemoteAppRoot/agent_plan/agent/server/services/predict_service.py"
    },
    @{
        Local = "D:\yolodo2.0\agent_plan\agent\server\tools\predict_tools.py"
        Remote = "$Server`:$RemoteAppRoot/agent_plan/agent/server/tools/predict_tools.py"
    },
    @{
        Local = "D:\yolodo2.0\agent_plan\agent\tests\test_prediction_remote_real_media.py"
        Remote = "$Server`:$RemoteAppRoot/agent_plan/agent/tests/test_prediction_remote_real_media.py"
    }
)

Write-Host "==> sync remote prediction code"
foreach ($item in $syncItems) {
    Invoke-NativeChecked -Exe "scp" -Args @(
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=10",
        $item.Local,
        $item.Remote
    )
}

$manifest = Join-Path $LocalStageRoot "manifest.json"
Write-Host "==> upload manifest"
Invoke-NativeChecked -Exe "scp" -Args @(
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=10",
    $manifest,
    "$Server`:$RemoteStageRoot/manifest.json"
)

Write-Host "==> upload weights"
Get-ChildItem -LiteralPath (Join-Path $LocalStageRoot "weights") -File | ForEach-Object {
    Invoke-NativeChecked -Exe "scp" -Args @(
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=10",
        $_.FullName,
        "$Server`:$RemoteStageRoot/weights/$($_.Name)"
    )
}

Write-Host "==> upload videos"
Get-ChildItem -LiteralPath (Join-Path $LocalStageRoot "videos") -File | ForEach-Object {
    Invoke-NativeChecked -Exe "scp" -Args @(
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=10",
        $_.FullName,
        "$Server`:$RemoteStageRoot/videos/$($_.Name)"
    )
}

Write-Host "==> run remote prediction validation"
$remoteCommand = @"
set -euo pipefail
source \$HOME/miniconda3/etc/profile.d/conda.sh
conda activate $EnvName
cd $RemoteAppRoot
python -m agent_plan.agent.tests.test_prediction_remote_real_media --weights-dir $RemoteStageRoot/weights --videos-dir $RemoteStageRoot/videos --output-dir $RemoteOutputRoot
"@
Invoke-NativeChecked -Exe "ssh" -Args @(
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=10",
    $Server,
    $remoteCommand
)

Write-Host "==> fetch remote validation result"
$localResultDir = Split-Path -Parent $LocalResultPath
New-Item -ItemType Directory -Force -Path $localResultDir | Out-Null
Invoke-NativeChecked -Exe "scp" -Args @(
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=10",
    "$Server`:$RemoteOutputRoot/remote_prediction_validation.json",
    $LocalResultPath
)

Write-Host "remote prediction roundtrip finished"
Write-Host "result: $LocalResultPath"
