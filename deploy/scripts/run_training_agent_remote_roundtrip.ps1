param(
    [string]$Server = "remote-agent",
    [ValidateSet("auto", "agent-server", "yolodo", "yolo")]
    [string]$EnvName = "auto",
    [string]$RemoteAppRoot = "",
    [string]$RemoteOutputRoot = "/tmp/training_real_lifecycle_output/agent_mainline_roundtrip",
    [string]$LocalResultPath = "",
    [string]$DatasetRoot = "",
    [string]$ModelPath = "",
    [int]$Epochs = 30,
    [int]$TargetEpoch = 2,
    [string]$StatusDelays = "15,35,60",
    [int]$ExtraPollInterval = 30,
    [int]$ExtraPollLimit = 8
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
$RepoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
. (Join-Path $PSScriptRoot "remote_script_common.ps1")

if ([string]::IsNullOrWhiteSpace($LocalResultPath)) {
    $LocalResultPath = Join-Path $RepoRoot "agent\tests\test_zyb_training_mainline_agent_roundtrip_output.json"
}

if ([string]::IsNullOrWhiteSpace($RemoteAppRoot)) {
    $RemoteAppRoot = Resolve-RemoteAppRoot -ServerName $Server
    Write-Host "resolved remote app root: $RemoteAppRoot"
}
if ([string]::IsNullOrWhiteSpace($DatasetRoot)) {
    $DatasetRoot = Resolve-RemoteDatasetRoot -ServerName $Server
    Write-Host "resolved remote dataset root: $DatasetRoot"
}
if ([string]::IsNullOrWhiteSpace($ModelPath)) {
    $ModelPath = Resolve-RemoteModelPath -ServerName $Server
    Write-Host "resolved remote model path: $ModelPath"
}

Write-Host "==> sync managed server_proto mirror to remote"
Sync-ManagedServerProtoMirror -Server $Server -RemoteAppRoot $RemoteAppRoot

Write-Host "==> ensure remote validation directories"
Ensure-RemoteDirectories -Server $Server -Commands @(
    "mkdir -p $RemoteAppRoot/deploy/scripts && echo __REMOTE_READY__ deploy_scripts",
    "mkdir -p $RemoteOutputRoot && echo __REMOTE_READY__ output"
)

Write-Host "==> sync remote training validation script"
Sync-RemoteTextFileNormalized -Server $Server `
    -LocalPath (Join-Path $RepoRoot "deploy\scripts\run_training_agent_remote_validation.sh") `
    -RemotePath "$RemoteAppRoot/deploy/scripts/run_training_agent_remote_validation.sh"

Write-Host "==> ensure remote mcp"
Invoke-RemoteSsh -Server $Server -Command "$RemoteAppRoot/manage_mcp_server.sh restart"

Write-Host "==> run remote training agent validation"
$remoteCommand = "bash $RemoteAppRoot/deploy/scripts/run_training_agent_remote_validation.sh $EnvName $RemoteOutputRoot $DatasetRoot $ModelPath $Epochs $TargetEpoch $StatusDelays $ExtraPollInterval $ExtraPollLimit"
Invoke-RemoteSsh -Server $Server -Command $remoteCommand

Write-Host "==> fetch remote validation result"
Fetch-RemoteFile -Server $Server -RemotePath "$RemoteOutputRoot/remote_training_mainline_agent_roundtrip.json" -LocalPath $LocalResultPath

Write-Host "remote training agent roundtrip finished"
Write-Host "result: $LocalResultPath"
