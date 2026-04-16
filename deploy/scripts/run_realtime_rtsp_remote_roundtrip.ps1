param(
    [Parameter(Mandatory = $true)]
    [string]$RtspUrl,

    [Parameter(Mandatory = $true)]
    [string]$ModelPath,

    [string]$Server = "yolostudio",

    [ValidateSet("auto", "yolo", "yolodo")]
    [string]$EnvName = "auto",

    [string]$RemoteAppRoot = "",
    [string]$RemoteOutputRoot = "/tmp/realtime_rtsp_validation",
    [string]$LocalResultPath = "",
    [string]$BashExe = "",
    [string]$SshExe = "",
    [string]$ScpExe = "",
    [int]$TimeoutMs = 5000,
    [int]$FrameIntervalMs = 120,
    [int]$MaxFrames = 8,
    [double]$WaitSeconds = 20,
    [double]$PollIntervalSeconds = 0.5
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
$RepoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
. (Join-Path $PSScriptRoot "remote_script_common.ps1")

if ([string]::IsNullOrWhiteSpace($LocalResultPath)) {
    $LocalResultPath = Join-Path $RepoRoot "agent\tests\test_realtime_rtsp_external_output.json"
}

$ResolvedBashExe = $null
try {
    $ResolvedBashExe = Resolve-NativeExecutable -Name 'bash' -ExplicitPath $BashExe
}
catch {
    $ResolvedBashExe = $null
}
$ResolvedSshExe = Resolve-NativeExecutable -Name 'ssh' -ExplicitPath $SshExe
$ResolvedScpExe = Resolve-NativeExecutable -Name 'scp' -ExplicitPath $ScpExe

Write-Host "using bash: $ResolvedBashExe"
Write-Host "using ssh : $ResolvedSshExe"
Write-Host "using scp : $ResolvedScpExe"

if ([string]::IsNullOrWhiteSpace($RemoteAppRoot)) {
    $RemoteAppRoot = Resolve-RemoteAppRoot -ServerName $Server -SshExe $ResolvedSshExe -BashExe $ResolvedBashExe
    Write-Host "resolved remote app root: $RemoteAppRoot"
}

Write-Host "==> sync managed server_proto mirror to remote"
Sync-ManagedServerProtoMirror -Server $Server -RemoteAppRoot $RemoteAppRoot -SshExe $ResolvedSshExe -ScpExe $ResolvedScpExe -BashExe $ResolvedBashExe

Write-Host "==> ensure remote directories"
Ensure-RemoteDirectories -Server $Server -SshExe $ResolvedSshExe -BashExe $ResolvedBashExe -Commands @(
    "mkdir -p $RemoteAppRoot/deploy/scripts",
    "mkdir -p $RemoteOutputRoot"
)

Write-Host "==> sync remote RTSP validation script"
Sync-RemoteTextFileNormalized -Server $Server `
    -LocalPath (Join-Path $RepoRoot "deploy\scripts\run_realtime_rtsp_remote_validation.sh") `
    -RemotePath "$RemoteAppRoot/deploy/scripts/run_realtime_rtsp_remote_validation.sh" `
    -SshExe $ResolvedSshExe `
    -ScpExe $ResolvedScpExe `
    -BashExe $ResolvedBashExe

Write-Host "==> run remote RTSP validation"
$remoteCommand = "bash $RemoteAppRoot/deploy/scripts/run_realtime_rtsp_remote_validation.sh " +
    "'$EnvName' '$RtspUrl' '$ModelPath' '$RemoteOutputRoot' '$TimeoutMs' '$FrameIntervalMs' '$MaxFrames' '$WaitSeconds' '$PollIntervalSeconds'"
Invoke-RemoteSsh -Server $Server -Command $remoteCommand -SshExe $ResolvedSshExe -BashExe $ResolvedBashExe

Write-Host "==> fetch remote validation result"
Fetch-RemoteFile -Server $Server -RemotePath "$RemoteOutputRoot/external_rtsp_validation.json" -LocalPath $LocalResultPath -ScpExe $ResolvedScpExe -BashExe $ResolvedBashExe

Write-Host "remote RTSP validation finished"
Write-Host "result: $LocalResultPath"
