param(
    [string]$PageUrl = "https://www.skylinewebcams.com/zh/webcam/thailand/surat-thani/ko-samui/lamai.html",
    [string]$Server = "yolostudio",
    [string]$RemoteAppRoot = '$HOME/yolostudio_agent_proto',
    [string]$RemoteWorkDir = "/tmp/skyline_rtsp_bridge",
    [string]$PathName = "skyline",
    [int]$RtspPort = 8555,
    [string]$ResultJson = "",
    [string]$BashExe = "",
    [string]$SshExe = "",
    [string]$ScpExe = ""
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
$RepoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))

if ([string]::IsNullOrWhiteSpace($ResultJson)) {
    $ResultJson = Join-Path $RepoRoot "agent\tests\skyline_rtsp_bridge_output.json"
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
        foreach ($candidate in @('C:\Program Files\Git\usr\bin\ssh.exe')) {
            if (-not $candidates.Contains($candidate)) { $candidates.Add($candidate) }
        }
    }
    elseif ($Name -eq 'scp') {
        foreach ($candidate in @('C:\Program Files\Git\usr\bin\scp.exe')) {
            if (-not $candidates.Contains($candidate)) { $candidates.Add($candidate) }
        }
    }
    elseif ($Name -eq 'bash') {
        foreach ($candidate in @('C:\Program Files\Git\usr\bin\bash.exe')) {
            if (-not $candidates.Contains($candidate)) { $candidates.Add($candidate) }
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

    throw "未找到可用的 $Name 可执行文件。"
}

function ConvertTo-BashLiteral {
    param([Parameter(Mandatory = $true)][string]$Value)
    $escaped = $Value.Replace("'", "'""'""'")
    return "'" + $escaped + "'"
}

function Invoke-BashWrappedNative {
    param(
        [Parameter(Mandatory = $true)][string]$Exe,
        [Parameter(Mandatory = $true)][string[]]$Args
    )

    $parts = @((ConvertTo-BashLiteral -Value $Exe)) + @($Args | ForEach-Object { ConvertTo-BashLiteral -Value $_ })
    $command = $parts -join ' '
    Write-Host "> $Exe $($Args -join ' ')"
    & $ResolvedBashExe -lc $command
    if ($LASTEXITCODE -ne 0) {
        throw "$Exe 执行失败，exit code=$LASTEXITCODE"
    }
}

$ResolvedBashExe = Resolve-NativeExecutable -Name 'bash' -ExplicitPath $BashExe
$ResolvedSshExe = Resolve-NativeExecutable -Name 'ssh' -ExplicitPath $SshExe
$ResolvedScpExe = Resolve-NativeExecutable -Name 'scp' -ExplicitPath $ScpExe

$bridgeScriptLocal = Join-Path $RepoRoot "deploy\scripts\run_dynamic_hls_rtsp_bridge_remote.sh"
$bridgeScriptRemote = "$RemoteAppRoot/deploy/scripts/run_dynamic_hls_rtsp_bridge_remote.sh"
$skylineScriptLocal = Join-Path $RepoRoot "deploy\scripts\run_skyline_rtsp_remote_bridge.sh"
$skylineScriptRemote = "$RemoteAppRoot/deploy/scripts/run_skyline_rtsp_remote_bridge.sh"

Write-Host "==> sync remote Skyline scripts"
Invoke-BashWrappedNative -Exe $ResolvedSshExe -Args @("-n", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", $Server, "mkdir -p $RemoteAppRoot/deploy/scripts")
Invoke-BashWrappedNative -Exe $ResolvedScpExe -Args @("-o", "BatchMode=yes", "-o", "ConnectTimeout=10", $bridgeScriptLocal, "$Server`:$bridgeScriptRemote")
Invoke-BashWrappedNative -Exe $ResolvedScpExe -Args @("-o", "BatchMode=yes", "-o", "ConnectTimeout=10", $skylineScriptLocal, "$Server`:$skylineScriptRemote")

Write-Host "==> start remote Skyline RTSP bridge"
$remoteCommand = "bash $skylineScriptRemote " +
    "'$PageUrl' '$RemoteWorkDir' '$RtspPort' '$PathName' '8002' '8003' '$bridgeScriptRemote'"
$bridgeOutput = & $ResolvedBashExe -lc @(
    ((ConvertTo-BashLiteral -Value $ResolvedSshExe) + ' ' +
        ((@("-n", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", $Server, $remoteCommand) |
            ForEach-Object { ConvertTo-BashLiteral -Value $_ }) -join ' '))
) | Out-String
if ($LASTEXITCODE -ne 0) {
    throw "远端 Skyline RTSP bridge 启动失败"
}

$jsonLine = ($bridgeOutput -split "`r?`n" | Where-Object { $_.Trim().StartsWith("{") -and $_.Trim().EndsWith("}") } | Select-Object -Last 1)
if ([string]::IsNullOrWhiteSpace($jsonLine)) {
    throw "未从远端 bridge 输出中解析到 JSON 结果。原始输出: $bridgeOutput"
}

$bridgeResult = $jsonLine | ConvertFrom-Json
$resultPayload = [ordered]@{
    generated_at = Get-Date -Format "yyyy-MM-ddTHH:mm:ss"
    page_url = $PageUrl
    hls_url = $bridgeResult.hls_url
    rtsp_url = $bridgeResult.rtsp_url
    remote_work_dir = $bridgeResult.work_dir
    summary = "已在服务器侧完成 Skyline fresh HLS 抓取与 RTSP bridge 启动"
    bridge_result = $bridgeResult
}

$resultDir = Split-Path -Parent $ResultJson
New-Item -ItemType Directory -Force -Path $resultDir | Out-Null
$resultPayload | ConvertTo-Json -Depth 8 | Set-Content -Path $ResultJson -Encoding utf8

Write-Host "result json: $ResultJson"
Write-Host "rtsp url   : $($bridgeResult.rtsp_url)"
