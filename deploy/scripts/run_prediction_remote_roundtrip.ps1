param(
    [string]$Server = "remote-agent",
    [ValidateSet("auto", "yolo", "yolodo")]
    [string]$EnvName = "auto",
    [string]$RemoteAppRoot = "/opt/yolostudio-agent",
    [string]$RemoteStageRoot = "/data/prediction_real_media_stage",
    [string]$RemoteOutputRoot = "/tmp/prediction_real_media_output/codex_roundtrip",
    [string]$LocalStageRoot = "",
    [string]$LocalResultPath = ""
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
$RepoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
if ([string]::IsNullOrWhiteSpace($LocalStageRoot)) {
    $LocalStageRoot = Join-Path $RepoRoot ".tmp_prediction_real_media_stage"
}
if ([string]::IsNullOrWhiteSpace($LocalResultPath)) {
    $LocalResultPath = Join-Path $RepoRoot "agent\tests\test_prediction_remote_real_media_output.json"
}

function Invoke-NativeChecked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Exe,

        [Parameter(Mandatory = $true)]
        [string[]]$Args
    )

    $display = "> " + $Exe + " " + ($Args -join " ")
    Write-Host $display

    if ($Exe -in @("ssh", "scp")) {
        $quotedArgs = $Args | ForEach-Object { '"' + (($_ -replace '"', '\"')) + '"' }
        $cmdLine = '"' + $Exe + '" ' + ($quotedArgs -join " ") + ' < NUL'
        $cmdExe = if ($env:ComSpec) { $env:ComSpec } else { "C:\Windows\System32\cmd.exe" }
        & $cmdExe /c $cmdLine
    }
    else {
        & $Exe @Args
    }

    if ($LASTEXITCODE -ne 0) {
        throw "$Exe 执行失败，exit code=$LASTEXITCODE"
    }
}

$stageScript = Join-Path $RepoRoot "deploy\scripts\stage_prediction_real_media.py"
if (!(Test-Path -LiteralPath (Join-Path $LocalStageRoot "manifest.json"))) {
    Write-Host "==> stage real media locally"
    python $stageScript
    if ($LASTEXITCODE -ne 0) {
        throw "stage_prediction_real_media.py 执行失败"
    }
}

Write-Host "==> ensure remote directories"
$ensureCommands = @(
    "mkdir -p $RemoteAppRoot/agent_plan/agent/server/services && echo __REMOTE_READY__ services",
    "mkdir -p $RemoteAppRoot/agent_plan/agent/server/tools && echo __REMOTE_READY__ tools",
    "mkdir -p $RemoteAppRoot/agent_plan/agent/tests && echo __REMOTE_READY__ tests",
    "mkdir -p $RemoteAppRoot/deploy/scripts && echo __REMOTE_READY__ deploy_scripts",
    "mkdir -p $RemoteStageRoot/weights && echo __REMOTE_READY__ weights",
    "mkdir -p $RemoteStageRoot/videos && echo __REMOTE_READY__ videos",
    "mkdir -p $RemoteOutputRoot && echo __REMOTE_READY__ output"
)
foreach ($remoteCommand in $ensureCommands) {
    Invoke-NativeChecked -Exe "ssh" -Args @("-n", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", $Server, $remoteCommand)
}

$syncItems = @(
    @{ Local = (Join-Path $RepoRoot "agent\server\services\predict_service.py"); Remote = "$Server`:$RemoteAppRoot/agent_plan/agent/server/services/predict_service.py" },
    @{ Local = (Join-Path $RepoRoot "agent\server\tools\predict_tools.py"); Remote = "$Server`:$RemoteAppRoot/agent_plan/agent/server/tools/predict_tools.py" },
    @{ Local = (Join-Path $RepoRoot "agent\tests\test_prediction_remote_real_media.py"); Remote = "$Server`:$RemoteAppRoot/agent_plan/agent/tests/test_prediction_remote_real_media.py" },
    @{ Local = (Join-Path $RepoRoot "deploy\scripts\run_prediction_remote_validation.sh"); Remote = "$Server`:$RemoteAppRoot/deploy/scripts/run_prediction_remote_validation.sh" }
)

Write-Host "==> sync remote prediction code"
foreach ($item in $syncItems) {
    Invoke-NativeChecked -Exe "scp" -Args @("-o", "BatchMode=yes", "-o", "ConnectTimeout=10", $item.Local, $item.Remote)
}

$manifest = Join-Path $LocalStageRoot "manifest.json"
Write-Host "==> upload manifest"
Invoke-NativeChecked -Exe "scp" -Args @("-o", "BatchMode=yes", "-o", "ConnectTimeout=10", $manifest, "$Server`:$RemoteStageRoot/manifest.json")

Write-Host "==> upload weights"
Get-ChildItem -LiteralPath (Join-Path $LocalStageRoot "weights") -File | ForEach-Object {
    Invoke-NativeChecked -Exe "scp" -Args @("-o", "BatchMode=yes", "-o", "ConnectTimeout=10", $_.FullName, "$Server`:$RemoteStageRoot/weights/$($_.Name)")
}

Write-Host "==> upload videos"
Get-ChildItem -LiteralPath (Join-Path $LocalStageRoot "videos") -File | ForEach-Object {
    Invoke-NativeChecked -Exe "scp" -Args @("-o", "BatchMode=yes", "-o", "ConnectTimeout=10", $_.FullName, "$Server`:$RemoteStageRoot/videos/$($_.Name)")
}

Write-Host "==> run remote prediction validation"
$remoteCommand = "bash $RemoteAppRoot/deploy/scripts/run_prediction_remote_validation.sh $EnvName $RemoteStageRoot $RemoteOutputRoot"
Invoke-NativeChecked -Exe "ssh" -Args @("-n", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", $Server, $remoteCommand)

Write-Host "==> fetch remote validation result"
$localResultDir = Split-Path -Parent $LocalResultPath
New-Item -ItemType Directory -Force -Path $localResultDir | Out-Null
Invoke-NativeChecked -Exe "scp" -Args @("-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "$Server`:$RemoteOutputRoot/remote_prediction_validation.json", $LocalResultPath)

Write-Host "remote prediction roundtrip finished"
Write-Host "result: $LocalResultPath"
