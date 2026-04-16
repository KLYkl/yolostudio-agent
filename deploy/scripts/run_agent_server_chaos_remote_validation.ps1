param(
    [string]$Server = "yolostudio",
    [ValidateSet("auto", "yolostudio-agent-server", "agent-server", "yolodo", "yolo")]
    [string]$EnvName = "auto",
    [string]$RemoteAppRoot = "",
    [string]$RemoteOutputRoot = "/tmp/agent_server_chaos_output",
    [string]$LocalSummaryPath = ""
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
$RepoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
. (Join-Path $PSScriptRoot "remote_script_common.ps1")

if ([string]::IsNullOrWhiteSpace($LocalSummaryPath)) {
    $LocalSummaryPath = Join-Path $RepoRoot "agent\tests\agent_server_chaos_remote_summary.json"
}
if ([string]::IsNullOrWhiteSpace($RemoteAppRoot)) {
    $RemoteAppRoot = Resolve-RemoteAppRoot -ServerName $Server
    Write-Host "resolved remote app root: $RemoteAppRoot"
}

Write-Host "==> sync managed server_proto mirror to remote"
Sync-ManagedServerProtoMirror -Server $Server -RemoteAppRoot $RemoteAppRoot

Write-Host "==> ensure remote validation directories"
Ensure-RemoteDirectories -Server $Server -Commands @(
    "mkdir -p $RemoteAppRoot/deploy/scripts && echo __REMOTE_READY__ deploy_scripts",
    "mkdir -p $RemoteOutputRoot && echo __REMOTE_READY__ output"
)

Write-Host "==> sync remote chaos validation script"
Sync-RemoteTextFileNormalized -Server $Server `
    -LocalPath (Join-Path $RepoRoot "deploy\scripts\run_agent_server_chaos_remote_validation.sh") `
    -RemotePath "$RemoteAppRoot/deploy/scripts/run_agent_server_chaos_remote_validation.sh"

Write-Host "==> run remote agent server chaos validation"
$remoteCommand = "bash $RemoteAppRoot/deploy/scripts/run_agent_server_chaos_remote_validation.sh $EnvName $RemoteOutputRoot"
Invoke-RemoteSsh -Server $Server -Command $remoteCommand

Write-Host "==> fetch remote chaos summary"
Fetch-RemoteFile -Server $Server -RemotePath "$RemoteOutputRoot/agent_server_chaos_summary.json" -LocalPath $LocalSummaryPath

Write-Host "remote agent server chaos validation finished"
Write-Host "summary: $LocalSummaryPath"
