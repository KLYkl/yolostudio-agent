param(
    [string]$Server = "remote-agent",
    [ValidateSet("auto", "yolostudio-agent-server", "agent-server", "yolodo", "yolo")]
    [string]$EnvName = "auto",
    [string]$RemoteAppRoot = "/opt/yolostudio-agent",
    [string]$RemoteOutputRoot = "/tmp/training_real_lifecycle_output/agent_followup_matrix",
    [string]$DatasetRoot = "/data/example_dataset",
    [string]$ModelPath = "/models/yolov8n.pt",
    [string]$LocalSummaryPath = ""
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
$RepoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
if ([string]::IsNullOrWhiteSpace($LocalSummaryPath)) {
    $LocalSummaryPath = Join-Path $RepoRoot "agent\tests\test_zyb_training_mainline_followup_matrix_output.json"
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
    "mkdir -p $RemoteAppRoot/agent_plan/agent/tests && echo __REMOTE_READY__ tests",
    "mkdir -p $RemoteAppRoot/deploy/scripts && echo __REMOTE_READY__ deploy_scripts",
    "mkdir -p $RemoteOutputRoot && echo __REMOTE_READY__ output"
)

foreach ($remoteCommand in $ensureCommands) {
    Invoke-NativeChecked -Exe "ssh" -Args @("-n", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", $Server, $remoteCommand)
}

$syncItems = @(
    @{ Local = (Join-Path $RepoRoot "agent\tests\test_zyb_training_mainline_agent_roundtrip.py"); Remote = "$Server`:$RemoteAppRoot/agent_plan/agent/tests/test_zyb_training_mainline_agent_roundtrip.py" },
    @{ Local = (Join-Path $RepoRoot "deploy\scripts\run_training_agent_remote_validation.sh"); Remote = "$Server`:$RemoteAppRoot/deploy/scripts/run_training_agent_remote_validation.sh" },
    @{ Local = (Join-Path $RepoRoot "deploy\scripts\run_training_agent_remote_followup_matrix.sh"); Remote = "$Server`:$RemoteAppRoot/deploy/scripts/run_training_agent_remote_followup_matrix.sh" }
)

foreach ($item in $syncItems) {
    Invoke-NativeChecked -Exe "scp" -Args @("-o", "BatchMode=yes", "-o", "ConnectTimeout=10", $item.Local, $item.Remote)
}

$remoteCommand = "bash $RemoteAppRoot/deploy/scripts/run_training_agent_remote_followup_matrix.sh $EnvName $RemoteOutputRoot $DatasetRoot $ModelPath"
Invoke-NativeChecked -Exe "ssh" -Args @("-n", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", $Server, $remoteCommand)

$localResultDir = Split-Path -Parent $LocalSummaryPath
New-Item -ItemType Directory -Force -Path $localResultDir | Out-Null
Invoke-NativeChecked -Exe "scp" -Args @("-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "$Server`:$RemoteOutputRoot/remote_training_mainline_followup_matrix.json", $LocalSummaryPath)

Write-Host "remote training follow-up matrix finished"
Write-Host "summary: $LocalSummaryPath"
