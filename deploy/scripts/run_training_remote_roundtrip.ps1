param(
    [string]$Server = "yolostudio",
    [ValidateSet("auto", "yolostudio-agent-server", "yolodo", "yolo")]
    [string]$EnvName = "auto",
    [string]$RemoteAppRoot = "/home/kly/yolostudio_agent_proto",
    [string]$RemoteOutputRoot = "/home/kly/training_real_lifecycle_output/codex_roundtrip",
    [string]$LocalResultPath = "D:\yolodo2.0\agent_plan\agent\tests\test_zyb_long_training_lifecycle_output.json",
    [string]$DatasetRoot = "/home/kly/agent_cap_tests/zyb",
    [string]$ModelPath = "/home/kly/yolov8n.pt",
    [int]$Epochs = 30,
    [int]$TargetEpoch = 2,
    [string]$StatusDelays = "15,35,60",
    [int]$ExtraPollInterval = 30,
    [int]$ExtraPollLimit = 8
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

    $display = "> " + $Exe + " " + ($Args -join " ")
    Write-Host $display

    if ($Exe -in @("ssh", "scp")) {
        $quotedArgs = $Args | ForEach-Object {
            '"' + (($_ -replace '"', '\"')) + '"'
        }
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
    "mkdir -p $RemoteAppRoot/agent_plan/agent/server/services && echo __REMOTE_READY__ services",
    "mkdir -p $RemoteAppRoot/agent_plan/agent/server/tools && echo __REMOTE_READY__ tools",
    "mkdir -p $RemoteAppRoot/agent_plan/agent/tests && echo __REMOTE_READY__ tests",
    "mkdir -p $RemoteAppRoot/deploy/scripts && echo __REMOTE_READY__ deploy_scripts",
    "mkdir -p $RemoteOutputRoot && echo __REMOTE_READY__ output"
)

Write-Host "==> ensure remote directories"
foreach ($remoteCommand in $ensureCommands) {
    Invoke-NativeChecked -Exe "ssh" -Args @(
        "-n",
        "-T",
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=10",
        $Server,
        $remoteCommand
    )
}

$syncItems = @(
    @{
        Local = "D:\yolodo2.0\agent_plan\agent\server\services\knowledge_service.py"
        Remote = "$Server`:$RemoteAppRoot/agent_plan/agent/server/services/knowledge_service.py"
    },
    @{
        Local = "D:\yolodo2.0\agent_plan\agent\server\mcp_server.py"
        Remote = "$Server`:$RemoteAppRoot/agent_plan/agent/server/mcp_server.py"
    },
    @{
        Local = "D:\yolodo2.0\agent_plan\agent\server\services\train_log_parser.py"
        Remote = "$Server`:$RemoteAppRoot/agent_plan/agent/server/services/train_log_parser.py"
    },
    @{
        Local = "D:\yolodo2.0\agent_plan\agent\server\services\train_service.py"
        Remote = "$Server`:$RemoteAppRoot/agent_plan/agent/server/services/train_service.py"
    },
    @{
        Local = "D:\yolodo2.0\agent_plan\agent\server\services\training_result_helpers.py"
        Remote = "$Server`:$RemoteAppRoot/agent_plan/agent/server/services/training_result_helpers.py"
    },
    @{
        Local = "D:\yolodo2.0\agent_plan\agent\server\tools\combo_tools.py"
        Remote = "$Server`:$RemoteAppRoot/agent_plan/agent/server/tools/combo_tools.py"
    },
    @{
        Local = "D:\yolodo2.0\agent_plan\agent\server\tools\data_tools.py"
        Remote = "$Server`:$RemoteAppRoot/agent_plan/agent/server/tools/data_tools.py"
    },
    @{
        Local = "D:\yolodo2.0\agent_plan\agent\server\tools\knowledge_tools.py"
        Remote = "$Server`:$RemoteAppRoot/agent_plan/agent/server/tools/knowledge_tools.py"
    },
    @{
        Local = "D:\yolodo2.0\agent_plan\agent\server\tools\train_tools.py"
        Remote = "$Server`:$RemoteAppRoot/agent_plan/agent/server/tools/train_tools.py"
    },
    @{
        Local = "D:\yolodo2.0\agent_plan\agent\tests\test_zyb_long_training_lifecycle.py"
        Remote = "$Server`:$RemoteAppRoot/agent_plan/agent/tests/test_zyb_long_training_lifecycle.py"
    },
    @{
        Local = "D:\yolodo2.0\agent_plan\deploy\scripts\run_training_remote_validation.sh"
        Remote = "$Server`:$RemoteAppRoot/deploy/scripts/run_training_remote_validation.sh"
    }
)

Write-Host "==> sync remote training code"
foreach ($item in $syncItems) {
    Invoke-NativeChecked -Exe "scp" -Args @(
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=10",
        $item.Local,
        $item.Remote
    )
}

Write-Host "==> ensure remote mcp"
Invoke-NativeChecked -Exe "ssh" -Args @(
    "-n",
    "-T",
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=10",
    $Server,
    "$RemoteAppRoot/manage_mcp_server.sh status || $RemoteAppRoot/manage_mcp_server.sh restart"
)

Write-Host "==> run remote training validation"
$remoteCommand = "bash $RemoteAppRoot/deploy/scripts/run_training_remote_validation.sh $EnvName $RemoteOutputRoot $DatasetRoot $ModelPath $Epochs $TargetEpoch $StatusDelays $ExtraPollInterval $ExtraPollLimit"
Invoke-NativeChecked -Exe "ssh" -Args @(
    "-n",
    "-T",
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
    "$Server`:$RemoteOutputRoot/remote_training_lifecycle.json",
    $LocalResultPath
)

Write-Host "remote training roundtrip finished"
Write-Host "result: $LocalResultPath"
