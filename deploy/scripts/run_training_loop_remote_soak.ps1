param(
    [string]$Server = "remote-agent",
    [ValidateSet("auto", "yolostudio-agent-server", "agent-server", "yolodo", "yolo")]
    [string]$EnvName = "auto",
    [ValidateSet("auto", "yolostudio-agent-server", "agent-server", "yolodo", "yolo", "")]
    [string]$TrainingEnvName = "",
    [string]$RemoteAppRoot = "",
    [string]$RemoteOutputRoot = "/tmp/training_real_lifecycle_output/training_loop_soak",
    [string]$DataYaml = "",
    [string]$ModelPath = "",
    [int]$MaxRounds = 20,
    [int]$Epochs = 1,
    [string]$Device = "0",
    [ValidateSet("forced", "real")]
    [string]$KnowledgeMode = "forced",
    [string]$ForcedAction = "continue_observing",
    [string]$AllowedTuningParams = "none",
    [ValidateSet("review", "conservative_auto", "full_auto")]
    [string]$ManagedLevel = "full_auto",
    [int]$TimeoutSeconds = 0,
    [int]$LoopPollInterval = 5,
    [int]$WatchPollInterval = 5,
    [ValidateSet("terminal", "review_or_terminal")]
    [string]$WaitMode = "terminal",
    [int]$AutoResumeReviews = 0,
    [switch]$RecreateServiceOnReviewResume,
    [string]$RemoteProject = "",
    [string]$LocalResultPath = ""
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
$RepoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
. (Join-Path $PSScriptRoot "remote_script_common.ps1")

$today = Get-Date -Format "yyyyMMdd"
if ([string]::IsNullOrWhiteSpace($LocalResultPath)) {
    $LocalResultPath = Join-Path $RepoRoot "output\training_loop_validation\$today\training_loop_remote_soak_${MaxRounds}r.json"
}
if ([string]::IsNullOrWhiteSpace($RemoteProject)) {
    $RemoteProject = "$RemoteOutputRoot/runs"
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

Write-Host "==> sync remote training loop soak script"
Sync-RemoteTextFileNormalized -Server $Server `
    -LocalPath (Join-Path $RepoRoot "deploy\scripts\run_training_loop_remote_soak.sh") `
    -RemotePath "$RemoteAppRoot/deploy/scripts/run_training_loop_remote_soak.sh"

$remoteDataYaml = if ([string]::IsNullOrWhiteSpace($DataYaml)) { "" } else { $DataYaml }
$remoteModelPath = if ([string]::IsNullOrWhiteSpace($ModelPath)) { "" } else { $ModelPath }
$remoteTrainingEnv = if ([string]::IsNullOrWhiteSpace($TrainingEnvName)) { "" } else { $TrainingEnvName }
$resumeFlag = if ($RecreateServiceOnReviewResume.IsPresent) { 1 } else { 0 }
$remoteCommand = "bash $RemoteAppRoot/deploy/scripts/run_training_loop_remote_soak.sh $EnvName '$remoteTrainingEnv' $RemoteOutputRoot '$remoteDataYaml' '$remoteModelPath' $MaxRounds $Epochs $Device $KnowledgeMode $ForcedAction $AllowedTuningParams $ManagedLevel $TimeoutSeconds $LoopPollInterval $WatchPollInterval '$RemoteProject' $WaitMode $AutoResumeReviews $resumeFlag"

Write-Host "==> run remote training loop soak"
Invoke-RemoteSsh -Server $Server -Command $remoteCommand

Write-Host "==> fetch remote soak result"
Fetch-RemoteFile -Server $Server -RemotePath "$RemoteOutputRoot/training_loop_remote_soak_${MaxRounds}r.json" -LocalPath $LocalResultPath

Write-Host "remote training loop soak finished"
Write-Host "result: $LocalResultPath"
