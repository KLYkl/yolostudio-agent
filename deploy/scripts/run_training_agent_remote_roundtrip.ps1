param(
    [string]$Server = "remote-agent",
    [ValidateSet("auto", "agent-server", "yolodo", "yolo")]
    [string]$EnvName = "auto",
    [string]$RemoteAppRoot = "/opt/yolostudio-agent",
    [string]$RemoteOutputRoot = "/tmp/training_real_lifecycle_output/agent_mainline_roundtrip",
    [string]$LocalResultPath = "",
    [string]$DatasetRoot = "/data/example_dataset",
    [string]$ModelPath = "/models/yolov8n.pt",
    [int]$Epochs = 30,
    [int]$TargetEpoch = 2,
    [string]$StatusDelays = "15,35,60",
    [int]$ExtraPollInterval = 30,
    [int]$ExtraPollLimit = 8
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
$RepoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
if ([string]::IsNullOrWhiteSpace($LocalResultPath)) {
    $LocalResultPath = Join-Path $RepoRoot "agent\tests\test_zyb_training_mainline_agent_roundtrip_output.json"
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
    "mkdir -p $RemoteAppRoot/agent_plan/agent/client && echo __REMOTE_READY__ client",
    "mkdir -p $RemoteAppRoot/agent_plan/agent/server/services && echo __REMOTE_READY__ services",
    "mkdir -p $RemoteAppRoot/agent_plan/agent/server/tools && echo __REMOTE_READY__ tools",
    "mkdir -p $RemoteAppRoot/agent_plan/agent/tests && echo __REMOTE_READY__ tests",
    "mkdir -p $RemoteAppRoot/deploy/scripts && echo __REMOTE_READY__ deploy_scripts",
    "mkdir -p $RemoteOutputRoot && echo __REMOTE_READY__ output"
)

Write-Host "==> ensure remote directories"
foreach ($remoteCommand in $ensureCommands) {
    Invoke-NativeChecked -Exe "ssh" -Args @("-n", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", $Server, $remoteCommand)
}

$syncItems = @(
    @{ Local = (Join-Path $RepoRoot "agent\client\agent_client.py"); Remote = "$Server`:$RemoteAppRoot/agent_plan/agent/client/agent_client.py" },
    @{ Local = (Join-Path $RepoRoot "agent\client\intent_parsing.py"); Remote = "$Server`:$RemoteAppRoot/agent_plan/agent/client/intent_parsing.py" },
    @{ Local = (Join-Path $RepoRoot "agent\client\session_state.py"); Remote = "$Server`:$RemoteAppRoot/agent_plan/agent/client/session_state.py" },
    @{ Local = (Join-Path $RepoRoot "agent\client\state_applier.py"); Remote = "$Server`:$RemoteAppRoot/agent_plan/agent/client/state_applier.py" },
    @{ Local = (Join-Path $RepoRoot "agent\client\context_builder.py"); Remote = "$Server`:$RemoteAppRoot/agent_plan/agent/client/context_builder.py" },
    @{ Local = (Join-Path $RepoRoot "agent\client\grounded_reply_builder.py"); Remote = "$Server`:$RemoteAppRoot/agent_plan/agent/client/grounded_reply_builder.py" },
    @{ Local = (Join-Path $RepoRoot "agent\server\services\knowledge_service.py"); Remote = "$Server`:$RemoteAppRoot/agent_plan/agent/server/services/knowledge_service.py" },
    @{ Local = (Join-Path $RepoRoot "agent\server\services\train_log_parser.py"); Remote = "$Server`:$RemoteAppRoot/agent_plan/agent/server/services/train_log_parser.py" },
    @{ Local = (Join-Path $RepoRoot "agent\server\services\train_service.py"); Remote = "$Server`:$RemoteAppRoot/agent_plan/agent/server/services/train_service.py" },
    @{ Local = (Join-Path $RepoRoot "agent\server\services\training_result_helpers.py"); Remote = "$Server`:$RemoteAppRoot/agent_plan/agent/server/services/training_result_helpers.py" },
    @{ Local = (Join-Path $RepoRoot "agent\server\tools\combo_tools.py"); Remote = "$Server`:$RemoteAppRoot/agent_plan/agent/server/tools/combo_tools.py" },
    @{ Local = (Join-Path $RepoRoot "agent\server\tools\data_tools.py"); Remote = "$Server`:$RemoteAppRoot/agent_plan/agent/server/tools/data_tools.py" },
    @{ Local = (Join-Path $RepoRoot "agent\server\tools\knowledge_tools.py"); Remote = "$Server`:$RemoteAppRoot/agent_plan/agent/server/tools/knowledge_tools.py" },
    @{ Local = (Join-Path $RepoRoot "agent\server\tools\train_tools.py"); Remote = "$Server`:$RemoteAppRoot/agent_plan/agent/server/tools/train_tools.py" },
    @{ Local = (Join-Path $RepoRoot "agent\tests\test_zyb_training_mainline_agent_roundtrip.py"); Remote = "$Server`:$RemoteAppRoot/agent_plan/agent/tests/test_zyb_training_mainline_agent_roundtrip.py" },
    @{ Local = (Join-Path $RepoRoot "deploy\scripts\run_training_agent_remote_validation.sh"); Remote = "$Server`:$RemoteAppRoot/deploy/scripts/run_training_agent_remote_validation.sh" }
)

Write-Host "==> sync remote training agent roundtrip code"
foreach ($item in $syncItems) {
    Invoke-NativeChecked -Exe "scp" -Args @("-o", "BatchMode=yes", "-o", "ConnectTimeout=10", $item.Local, $item.Remote)
}

Write-Host "==> ensure remote mcp"
Invoke-NativeChecked -Exe "ssh" -Args @("-n", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", $Server, "$RemoteAppRoot/manage_mcp_server.sh status || $RemoteAppRoot/manage_mcp_server.sh restart")

Write-Host "==> run remote training agent validation"
$remoteCommand = "bash $RemoteAppRoot/deploy/scripts/run_training_agent_remote_validation.sh $EnvName $RemoteOutputRoot $DatasetRoot $ModelPath $Epochs $TargetEpoch $StatusDelays $ExtraPollInterval $ExtraPollLimit"
Invoke-NativeChecked -Exe "ssh" -Args @("-n", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", $Server, $remoteCommand)

Write-Host "==> fetch remote validation result"
$localResultDir = Split-Path -Parent $LocalResultPath
New-Item -ItemType Directory -Force -Path $localResultDir | Out-Null
Invoke-NativeChecked -Exe "scp" -Args @("-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "$Server`:$RemoteOutputRoot/remote_training_mainline_agent_roundtrip.json", $LocalResultPath)

Write-Host "remote training agent roundtrip finished"
Write-Host "result: $LocalResultPath"
