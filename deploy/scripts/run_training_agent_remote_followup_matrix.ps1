param(
    [string]$Server = "remote-agent",
    [ValidateSet("auto", "yolostudio-agent-server", "agent-server", "yolodo", "yolo")]
    [string]$EnvName = "auto",
    [string]$RemoteAppRoot = "",
    [string]$RemoteOutputRoot = "/tmp/training_real_lifecycle_output/agent_followup_matrix",
    [string]$DatasetRoot = "",
    [string]$ModelPath = "",
    [string]$LocalSummaryPath = ""
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
$RepoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
. (Join-Path $PSScriptRoot "remote_script_common.ps1")

if ([string]::IsNullOrWhiteSpace($LocalSummaryPath)) {
    $LocalSummaryPath = Join-Path $RepoRoot "agent\tests\test_zyb_training_mainline_followup_matrix_output.json"
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

Write-Host "==> sync remote follow-up scripts"
Sync-RemoteTextFileNormalized -Server $Server `
    -LocalPath (Join-Path $RepoRoot "deploy\scripts\run_training_agent_remote_validation.sh") `
    -RemotePath "$RemoteAppRoot/deploy/scripts/run_training_agent_remote_validation.sh"
Sync-RemoteTextFileNormalized -Server $Server `
    -LocalPath (Join-Path $RepoRoot "deploy\scripts\run_training_agent_remote_followup_matrix.sh") `
    -RemotePath "$RemoteAppRoot/deploy/scripts/run_training_agent_remote_followup_matrix.sh"

Write-Host "==> run remote training follow-up matrix"
$remoteCommand = "bash $RemoteAppRoot/deploy/scripts/run_training_agent_remote_followup_matrix.sh $EnvName $RemoteOutputRoot $DatasetRoot $ModelPath"
Invoke-RemoteSsh -Server $Server -Command $remoteCommand

Write-Host "==> fetch remote matrix result"
Fetch-RemoteFile -Server $Server -RemotePath "$RemoteOutputRoot/remote_training_mainline_followup_matrix.json" -LocalPath $LocalSummaryPath

Write-Host "remote training follow-up matrix finished"
Write-Host "summary: $LocalSummaryPath"
