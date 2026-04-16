param(
    [string]$Server = "remote-agent",
    [ValidateSet("auto", "yolo", "yolodo")]
    [string]$EnvName = "auto",
    [string]$RemoteAppRoot = "",
    [string]$RemoteStageRoot = "/data/prediction_real_media_stage",
    [string]$RemoteOutputRoot = "/tmp/prediction_real_media_output/codex_roundtrip",
    [string]$LocalStageRoot = "",
    [string]$LocalResultPath = ""
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
$RepoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
. (Join-Path $PSScriptRoot "remote_script_common.ps1")
$LocalPython = Join-Path $RepoRoot "agent\.venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $LocalPython)) {
    $LocalPython = "python"
}

if ([string]::IsNullOrWhiteSpace($LocalStageRoot)) {
    $LocalStageRoot = Join-Path $RepoRoot ".tmp_prediction_real_media_stage"
}
if ([string]::IsNullOrWhiteSpace($LocalResultPath)) {
    $LocalResultPath = Join-Path $RepoRoot "agent\tests\test_prediction_remote_real_media_output.json"
}
if ([string]::IsNullOrWhiteSpace($RemoteAppRoot)) {
    $RemoteAppRoot = Resolve-RemoteAppRoot -ServerName $Server
    Write-Host "resolved remote app root: $RemoteAppRoot"
}

$stageScript = Join-Path $RepoRoot "deploy\scripts\stage_prediction_real_media.py"
if (!(Test-Path -LiteralPath (Join-Path $LocalStageRoot "manifest.json"))) {
    Write-Host "==> stage real media locally"
    & $LocalPython $stageScript
    if ($LASTEXITCODE -ne 0) {
        throw "stage_prediction_real_media.py failed"
    }
}

Write-Host "==> sync managed server_proto mirror to remote"
Sync-ManagedServerProtoMirror -Server $Server -RemoteAppRoot $RemoteAppRoot

Write-Host "==> ensure remote directories"
Ensure-RemoteDirectories -Server $Server -Commands @(
    "mkdir -p $RemoteAppRoot/deploy/scripts && echo __REMOTE_READY__ deploy_scripts",
    "mkdir -p $RemoteStageRoot/weights && echo __REMOTE_READY__ weights",
    "mkdir -p $RemoteStageRoot/videos && echo __REMOTE_READY__ videos",
    "mkdir -p $RemoteOutputRoot && echo __REMOTE_READY__ output"
)

Write-Host "==> sync remote prediction validation script"
Sync-RemoteTextFileNormalized -Server $Server `
    -LocalPath (Join-Path $RepoRoot "deploy\scripts\run_prediction_remote_validation.sh") `
    -RemotePath "$RemoteAppRoot/deploy/scripts/run_prediction_remote_validation.sh"

Write-Host "==> upload manifest"
Sync-RemoteFiles -Items @(
    @{
        Local = (Join-Path $LocalStageRoot "manifest.json")
        Remote = "$Server`:$RemoteStageRoot/manifest.json"
    }
)

Write-Host "==> upload weights"
Get-ChildItem -LiteralPath (Join-Path $LocalStageRoot "weights") -File | ForEach-Object {
    Sync-RemoteFiles -Items @(
        @{
            Local = $_.FullName
            Remote = "$Server`:$RemoteStageRoot/weights/$($_.Name)"
        }
    )
}

Write-Host "==> upload videos"
Get-ChildItem -LiteralPath (Join-Path $LocalStageRoot "videos") -File | ForEach-Object {
    Sync-RemoteFiles -Items @(
        @{
            Local = $_.FullName
            Remote = "$Server`:$RemoteStageRoot/videos/$($_.Name)"
        }
    )
}

Write-Host "==> run remote prediction validation"
$remoteCommand = "bash $RemoteAppRoot/deploy/scripts/run_prediction_remote_validation.sh $EnvName $RemoteStageRoot $RemoteOutputRoot"
Invoke-RemoteSsh -Server $Server -Command $remoteCommand

Write-Host "==> fetch remote validation result"
Fetch-RemoteFile -Server $Server -RemotePath "$RemoteOutputRoot/remote_prediction_validation.json" -LocalPath $LocalResultPath

Write-Host "remote prediction roundtrip finished"
Write-Host "result: $LocalResultPath"
