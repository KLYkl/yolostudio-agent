param(
    [string]$Server = "remote-agent",
    [ValidateSet("auto", "yolostudio-agent-server", "agent-server", "yolodo", "yolo")]
    [string]$EnvName = "auto",
    [ValidateSet("auto", "yolostudio-agent-server", "agent-server", "yolodo", "yolo", "")]
    [string]$TrainingEnvName = "",
    [string]$RemoteAppRoot = "/opt/yolostudio-agent",
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
$today = Get-Date -Format "yyyyMMdd"
if ([string]::IsNullOrWhiteSpace($LocalResultPath)) {
    $LocalResultPath = Join-Path $RepoRoot "output\training_loop_validation\$today\training_loop_remote_soak_${MaxRounds}r.json"
}
if ([string]::IsNullOrWhiteSpace($RemoteProject)) {
    $RemoteProject = "$RemoteOutputRoot/runs"
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

$ensureCommands = @(
    "mkdir -p $RemoteAppRoot/agent_plan/yolostudio_agent && echo __REMOTE_READY__ pkg_bridge",
    "mkdir -p $RemoteAppRoot/agent_plan/agent/client && echo __REMOTE_READY__ client",
    "mkdir -p $RemoteAppRoot/agent_plan/agent/server && echo __REMOTE_READY__ server",
    "mkdir -p $RemoteAppRoot/agent_plan/agent/server/services && echo __REMOTE_READY__ services",
    "mkdir -p $RemoteAppRoot/agent_plan/agent/tests && echo __REMOTE_READY__ tests",
    "mkdir -p $RemoteAppRoot/agent_plan/knowledge/core && echo __REMOTE_READY__ knowledge_core",
    "mkdir -p $RemoteAppRoot/agent_plan/knowledge/families/yolo && echo __REMOTE_READY__ knowledge_yolo",
    "mkdir -p $RemoteAppRoot/agent_plan/knowledge/playbooks && echo __REMOTE_READY__ knowledge_playbooks",
    "mkdir -p $RemoteAppRoot/deploy/scripts && echo __REMOTE_READY__ deploy_scripts",
    "mkdir -p $RemoteOutputRoot && echo __REMOTE_READY__ output"
)

Write-Host "==> ensure remote directories"
foreach ($remoteCommand in $ensureCommands) {
    Invoke-NativeChecked -Exe "ssh" -Args @("-n", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", $Server, $remoteCommand)
}

$syncItems = @(
    @{ Local = (Join-Path $RepoRoot "yolostudio_agent\__init__.py"); Remote = "$Server`:$RemoteAppRoot/agent_plan/yolostudio_agent/__init__.py" },
    @{ Local = (Join-Path $RepoRoot "agent\__init__.py"); Remote = "$Server`:$RemoteAppRoot/agent_plan/agent/__init__.py" },
    @{ Local = (Join-Path $RepoRoot "agent\client\__init__.py"); Remote = "$Server`:$RemoteAppRoot/agent_plan/agent/client/__init__.py" },
    @{ Local = (Join-Path $RepoRoot "agent\client\llm_factory.py"); Remote = "$Server`:$RemoteAppRoot/agent_plan/agent/client/llm_factory.py" },
    @{ Local = (Join-Path $RepoRoot "agent\server\__init__.py"); Remote = "$Server`:$RemoteAppRoot/agent_plan/agent/server/__init__.py" },
    @{ Local = (Join-Path $RepoRoot "agent\server\services\__init__.py"); Remote = "$Server`:$RemoteAppRoot/agent_plan/agent/server/services/__init__.py" },
    @{ Local = (Join-Path $RepoRoot "agent\server\services\gpu_utils.py"); Remote = "$Server`:$RemoteAppRoot/agent_plan/agent/server/services/gpu_utils.py" },
    @{ Local = (Join-Path $RepoRoot "agent\server\services\knowledge_service.py"); Remote = "$Server`:$RemoteAppRoot/agent_plan/agent/server/services/knowledge_service.py" },
    @{ Local = (Join-Path $RepoRoot "agent\server\services\train_log_parser.py"); Remote = "$Server`:$RemoteAppRoot/agent_plan/agent/server/services/train_log_parser.py" },
    @{ Local = (Join-Path $RepoRoot "agent\server\services\train_service.py"); Remote = "$Server`:$RemoteAppRoot/agent_plan/agent/server/services/train_service.py" },
    @{ Local = (Join-Path $RepoRoot "agent\server\services\training_result_helpers.py"); Remote = "$Server`:$RemoteAppRoot/agent_plan/agent/server/services/training_result_helpers.py" },
    @{ Local = (Join-Path $RepoRoot "agent\server\services\training_loop_service.py"); Remote = "$Server`:$RemoteAppRoot/agent_plan/agent/server/services/training_loop_service.py" },
    @{ Local = (Join-Path $RepoRoot "agent\tests\__init__.py"); Remote = "$Server`:$RemoteAppRoot/agent_plan/agent/tests/__init__.py" },
    @{ Local = (Join-Path $RepoRoot "agent\tests\training_loop_soak_support.py"); Remote = "$Server`:$RemoteAppRoot/agent_plan/agent/tests/training_loop_soak_support.py" },
    @{ Local = (Join-Path $RepoRoot "agent\tests\test_training_loop_remote_real_soak.py"); Remote = "$Server`:$RemoteAppRoot/agent_plan/agent/tests/test_training_loop_remote_real_soak.py" },
    @{ Local = (Join-Path $RepoRoot "knowledge\index.json"); Remote = "$Server`:$RemoteAppRoot/agent_plan/knowledge/index.json" },
    @{ Local = (Join-Path $RepoRoot "knowledge\core\pre_training_rules.json"); Remote = "$Server`:$RemoteAppRoot/agent_plan/knowledge/core/pre_training_rules.json" },
    @{ Local = (Join-Path $RepoRoot "knowledge\core\post_training_rules.json"); Remote = "$Server`:$RemoteAppRoot/agent_plan/knowledge/core/post_training_rules.json" },
    @{ Local = (Join-Path $RepoRoot "knowledge\core\next_step_rules.json"); Remote = "$Server`:$RemoteAppRoot/agent_plan/knowledge/core/next_step_rules.json" },
    @{ Local = (Join-Path $RepoRoot "knowledge\families\yolo\detection_rules.json"); Remote = "$Server`:$RemoteAppRoot/agent_plan/knowledge/families/yolo/detection_rules.json" },
    @{ Local = (Join-Path $RepoRoot "knowledge\playbooks\training_preflight_basics.md"); Remote = "$Server`:$RemoteAppRoot/agent_plan/knowledge/playbooks/training_preflight_basics.md" },
    @{ Local = (Join-Path $RepoRoot "knowledge\playbooks\training_metrics_basics.md"); Remote = "$Server`:$RemoteAppRoot/agent_plan/knowledge/playbooks/training_metrics_basics.md" },
    @{ Local = (Join-Path $RepoRoot "knowledge\playbooks\yolo_detection_guidance.md"); Remote = "$Server`:$RemoteAppRoot/agent_plan/knowledge/playbooks/yolo_detection_guidance.md" },
    @{ Local = (Join-Path $RepoRoot "deploy\scripts\run_training_loop_remote_soak.sh"); Remote = "$Server`:$RemoteAppRoot/deploy/scripts/run_training_loop_remote_soak.sh" },
    @{ Local = (Join-Path $RepoRoot "deploy\scripts\run_training_loop_remote_soak.ps1"); Remote = "$Server`:$RemoteAppRoot/deploy/scripts/run_training_loop_remote_soak.ps1" }
)

Write-Host "==> sync remote training loop soak code"
foreach ($item in $syncItems) {
    Invoke-NativeChecked -Exe "scp" -Args @("-o", "BatchMode=yes", "-o", "ConnectTimeout=10", $item.Local, $item.Remote)
}

$remoteDataYaml = if ([string]::IsNullOrWhiteSpace($DataYaml)) { "" } else { $DataYaml }
$remoteModelPath = if ([string]::IsNullOrWhiteSpace($ModelPath)) { "" } else { $ModelPath }
$remoteTrainingEnv = if ([string]::IsNullOrWhiteSpace($TrainingEnvName)) { "" } else { $TrainingEnvName }
$resumeFlag = if ($RecreateServiceOnReviewResume.IsPresent) { 1 } else { 0 }
$remoteCommand = "bash $RemoteAppRoot/deploy/scripts/run_training_loop_remote_soak.sh $EnvName '$remoteTrainingEnv' $RemoteOutputRoot '$remoteDataYaml' '$remoteModelPath' $MaxRounds $Epochs $Device $KnowledgeMode $ForcedAction $AllowedTuningParams $ManagedLevel $TimeoutSeconds $LoopPollInterval $WatchPollInterval '$RemoteProject' $WaitMode $AutoResumeReviews $resumeFlag"

Write-Host "==> run remote training loop soak"
Invoke-NativeChecked -Exe "ssh" -Args @("-n", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", $Server, $remoteCommand)

Write-Host "==> fetch remote soak result"
$localResultDir = Split-Path -Parent $LocalResultPath
New-Item -ItemType Directory -Force -Path $localResultDir | Out-Null
Invoke-NativeChecked -Exe "scp" -Args @("-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "$Server`:$RemoteOutputRoot/training_loop_remote_soak_${MaxRounds}r.json", $LocalResultPath)

Write-Host "remote training loop soak finished"
Write-Host "result: $LocalResultPath"
