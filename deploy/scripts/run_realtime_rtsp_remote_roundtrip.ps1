param(
    [Parameter(Mandatory = $true)]
    [string]$RtspUrl,

    [Parameter(Mandatory = $true)]
    [string]$ModelPath,

    [string]$Server = "yolostudio",

    [ValidateSet("auto", "yolo", "yolodo")]
    [string]$EnvName = "auto",

    [string]$RemoteAppRoot = '$HOME/yolostudio_agent_proto',
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
if ([string]::IsNullOrWhiteSpace($LocalResultPath)) {
    $LocalResultPath = Join-Path $RepoRoot "agent\tests\test_realtime_rtsp_external_output.json"
}

function Resolve-NativeExecutable {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name,

        [string]$ExplicitPath = ""
    )

    $candidates = New-Object System.Collections.Generic.List[string]
    if (-not [string]::IsNullOrWhiteSpace($ExplicitPath)) {
        $candidates.Add($ExplicitPath)
    }

    if ($Name -eq 'ssh') {
        foreach ($candidate in @(
            'C:\Program Files\Git\usr\bin\ssh.exe'
        )) {
            if (-not $candidates.Contains($candidate)) {
                $candidates.Add($candidate)
            }
        }
    }
    elseif ($Name -eq 'scp') {
        foreach ($candidate in @(
            'C:\Program Files\Git\usr\bin\scp.exe'
        )) {
            if (-not $candidates.Contains($candidate)) {
                $candidates.Add($candidate)
            }
        }
    }
    elseif ($Name -eq 'bash') {
        foreach ($candidate in @(
            'C:\Program Files\Git\usr\bin\bash.exe'
        )) {
            if (-not $candidates.Contains($candidate)) {
                $candidates.Add($candidate)
            }
        }
    }

    $cmd = Get-Command $Name -ErrorAction SilentlyContinue
    if ($cmd -and -not $candidates.Contains($cmd.Source)) {
        $candidates.Add($cmd.Source)
    }

    foreach ($candidate in $candidates) {
        if (-not [string]::IsNullOrWhiteSpace($candidate) -and (Test-Path -LiteralPath $candidate)) {
            return $candidate
        }
    }

    throw "未找到可用的 $Name 可执行文件。已尝试: $($candidates -join ', ')"
}

function ConvertTo-BashLiteral {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Value
    )

    $escaped = $Value.Replace("'", "'""'""'")
    return "'" + $escaped + "'"
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

    $leaf = [System.IO.Path]::GetFileNameWithoutExtension($Exe)
    if ($ResolvedBashExe -and $leaf -in @('ssh', 'scp')) {
        $bashParts = @((ConvertTo-BashLiteral -Value $Exe)) + @($Args | ForEach-Object { ConvertTo-BashLiteral -Value $_ })
        $bashCommand = $bashParts -join ' '
        & $ResolvedBashExe -lc $bashCommand
    }
    elseif ($Exe -in @("ssh", "scp")) {
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

$ensureCommands = @(
    "mkdir -p $RemoteAppRoot/agent_plan/agent/server/services",
    "mkdir -p $RemoteAppRoot/agent_plan/agent/tests",
    "mkdir -p $RemoteAppRoot/deploy/scripts",
    "mkdir -p $RemoteOutputRoot"
)
foreach ($remoteCommand in $ensureCommands) {
    Invoke-NativeChecked -Exe $ResolvedSshExe -Args @("-n", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", $Server, $remoteCommand)
}

$syncItems = @(
    @{ Local = (Join-Path $RepoRoot "agent\server\services\predict_service.py"); Remote = "$Server`:$RemoteAppRoot/agent_plan/agent/server/services/predict_service.py" },
    @{ Local = (Join-Path $RepoRoot "agent\server\services\realtime_device_helpers.py"); Remote = "$Server`:$RemoteAppRoot/agent_plan/agent/server/services/realtime_device_helpers.py" },
    @{ Local = (Join-Path $RepoRoot "agent\server\services\realtime_predict_service.py"); Remote = "$Server`:$RemoteAppRoot/agent_plan/agent/server/services/realtime_predict_service.py" },
    @{ Local = (Join-Path $RepoRoot "agent\tests\test_realtime_rtsp_external_validation.py"); Remote = "$Server`:$RemoteAppRoot/agent_plan/agent/tests/test_realtime_rtsp_external_validation.py" },
    @{ Local = (Join-Path $RepoRoot "deploy\scripts\run_realtime_rtsp_remote_validation.sh"); Remote = "$Server`:$RemoteAppRoot/deploy/scripts/run_realtime_rtsp_remote_validation.sh" }
)

Write-Host "==> sync remote RTSP validation code"
foreach ($item in $syncItems) {
    Invoke-NativeChecked -Exe $ResolvedScpExe -Args @("-o", "BatchMode=yes", "-o", "ConnectTimeout=10", $item.Local, $item.Remote)
}

Write-Host "==> run remote RTSP validation"
$remoteCommand = "bash $RemoteAppRoot/deploy/scripts/run_realtime_rtsp_remote_validation.sh " +
    "'$EnvName' '$RtspUrl' '$ModelPath' '$RemoteOutputRoot' '$TimeoutMs' '$FrameIntervalMs' '$MaxFrames' '$WaitSeconds' '$PollIntervalSeconds'"
Invoke-NativeChecked -Exe $ResolvedSshExe -Args @("-n", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", $Server, $remoteCommand)

Write-Host "==> fetch remote validation result"
$localResultDir = Split-Path -Parent $LocalResultPath
New-Item -ItemType Directory -Force -Path $localResultDir | Out-Null
Invoke-NativeChecked -Exe $ResolvedScpExe -Args @("-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "$Server`:$RemoteOutputRoot/external_rtsp_validation.json", $LocalResultPath)

Write-Host "remote RTSP validation finished"
Write-Host "result: $LocalResultPath"
