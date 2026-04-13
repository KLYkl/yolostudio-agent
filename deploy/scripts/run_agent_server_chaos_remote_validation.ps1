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
if ([string]::IsNullOrWhiteSpace($LocalSummaryPath)) {
    $LocalSummaryPath = Join-Path $RepoRoot "agent\tests\agent_server_chaos_remote_summary.json"
}

function Resolve-RemoteAppRoot {
    param([string]$ServerName)

    $probe = @'
set -e
for candidate in "$PWD" "$HOME/yolostudio_agent_proto" "/opt/yolostudio-agent"; do
  if [ -d "$candidate/agent_plan/agent/tests" ]; then
    printf '%s\n' "$candidate"
    exit 0
  fi
done
printf '%s\n' "$HOME/yolostudio_agent_proto"
'@

    $result = & ssh -n -T -o BatchMode=yes -o ConnectTimeout=10 $ServerName $probe
    if ($LASTEXITCODE -ne 0) {
        throw "无法探测远端应用目录，exit code=$LASTEXITCODE"
    }

    $resolved = ($result | Select-Object -Last 1).Trim()
    if ([string]::IsNullOrWhiteSpace($resolved)) {
        throw "远端应用目录探测结果为空"
    }
    return $resolved
}

if ([string]::IsNullOrWhiteSpace($RemoteAppRoot)) {
    $RemoteAppRoot = Resolve-RemoteAppRoot -ServerName $Server
    Write-Host "resolved remote app root: $RemoteAppRoot"
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
    "mkdir -p $RemoteAppRoot/agent_plan/agent/client && echo __REMOTE_READY__ client",
    "mkdir -p $RemoteAppRoot/deploy/scripts && echo __REMOTE_READY__ deploy_scripts",
    "mkdir -p $RemoteOutputRoot && echo __REMOTE_READY__ output"
)
foreach ($remoteCommand in $ensureCommands) {
    Invoke-NativeChecked -Exe "ssh" -Args @("-n", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", $Server, $remoteCommand)
}

$syncItems = @(
    @{ Local = (Join-Path $RepoRoot "agent\client\agent_client.py"); Remote = "$Server`:$RemoteAppRoot/agent_plan/agent/client/agent_client.py" },
    @{ Local = (Join-Path $RepoRoot "agent\tests\_coroutine_runner.py"); Remote = "$Server`:$RemoteAppRoot/agent_plan/agent/tests/_coroutine_runner.py" },
    @{ Local = (Join-Path $RepoRoot "deploy\scripts\run_agent_server_chaos_remote_validation.sh"); Remote = "$Server`:$RemoteAppRoot/deploy/scripts/run_agent_server_chaos_remote_validation.sh" }
)
Get-ChildItem -Path (Join-Path $RepoRoot "agent\tests") -Filter "test_agent_server_chaos_*.py" | ForEach-Object {
    $syncItems += @{ Local = $_.FullName; Remote = "$Server`:$RemoteAppRoot/agent_plan/agent/tests/$($_.Name)" }
}
foreach ($item in $syncItems) {
    Invoke-NativeChecked -Exe "scp" -Args @("-o", "BatchMode=yes", "-o", "ConnectTimeout=10", $item.Local, $item.Remote)
}

$remoteCommand = "bash $RemoteAppRoot/deploy/scripts/run_agent_server_chaos_remote_validation.sh $EnvName $RemoteOutputRoot"
Invoke-NativeChecked -Exe "ssh" -Args @("-n", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", $Server, $remoteCommand)

$localResultDir = Split-Path -Parent $LocalSummaryPath
New-Item -ItemType Directory -Force -Path $localResultDir | Out-Null
Invoke-NativeChecked -Exe "scp" -Args @("-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "$Server`:$RemoteOutputRoot/agent_server_chaos_summary.json", $LocalSummaryPath)

Write-Host "remote agent server chaos validation finished"
Write-Host "summary: $LocalSummaryPath"
