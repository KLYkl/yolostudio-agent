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
            if (-not $candidates.Contains($candidate)) {
                $candidates.Add($candidate)
            }
        }
    }
    elseif ($Name -eq 'scp') {
        foreach ($candidate in @('C:\Program Files\Git\usr\bin\scp.exe')) {
            if (-not $candidates.Contains($candidate)) {
                $candidates.Add($candidate)
            }
        }
    }
    elseif ($Name -eq 'bash') {
        foreach ($candidate in @('C:\Program Files\Git\usr\bin\bash.exe')) {
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

    throw "No usable $Name executable found. Tried: $($candidates -join ', ')"
}


function ConvertTo-BashLiteral {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Value
    )

    $escaped = $Value.Replace("\\", "\\\\")
    $escaped = $escaped.Replace('"', '\"')
    $escaped = $escaped.Replace('$', '\$')
    return '"' + $escaped + '"'
}


function Invoke-NativeCapture {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Exe,

        [Parameter(Mandatory = $true)]
        [string[]]$Args,

        [string]$BashExe = ""
    )

    $display = "> " + $Exe + " " + ($Args -join " ")
    Write-Host $display

    $leaf = [System.IO.Path]::GetFileNameWithoutExtension($Exe)
    if (-not [string]::IsNullOrWhiteSpace($BashExe) -and $leaf -in @('ssh', 'scp')) {
        $bashParts = @((ConvertTo-BashLiteral -Value $Exe)) + @($Args | ForEach-Object { ConvertTo-BashLiteral -Value $_ })
        $bashCommand = $bashParts -join ' '
        $output = & $BashExe -lc $bashCommand
    }
    else {
        $output = & $Exe @Args
    }

    if ($LASTEXITCODE -ne 0) {
        throw "$Exe failed, exit code=$LASTEXITCODE"
    }

    return $output
}


function Invoke-NativeChecked {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Exe,

        [Parameter(Mandatory = $true)]
        [string[]]$Args,

        [string]$BashExe = ""
    )

    [void](Invoke-NativeCapture -Exe $Exe -Args $Args -BashExe $BashExe)
}


function Invoke-RemoteSsh {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Server,

        [Parameter(Mandatory = $true)]
        [string]$Command,

        [string]$SshExe = "ssh",
        [string]$BashExe = ""
    )

    $NormalizedCommand = $Command -replace "`r", ""
    Invoke-NativeChecked -Exe $SshExe -Args @("-n", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", $Server, $NormalizedCommand) -BashExe $BashExe
}


function Invoke-RemoteSshCapture {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Server,

        [Parameter(Mandatory = $true)]
        [string]$Command,

        [string]$SshExe = "ssh",
        [string]$BashExe = ""
    )

    $NormalizedCommand = $Command -replace "`r", ""
    return Invoke-NativeCapture -Exe $SshExe -Args @("-n", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", $Server, $NormalizedCommand) -BashExe $BashExe
}


function Ensure-RemoteDirectories {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Server,

        [Parameter(Mandatory = $true)]
        [string[]]$Commands,

        [string]$SshExe = "ssh",
        [string]$BashExe = ""
    )

    foreach ($remoteCommand in $Commands) {
        Invoke-RemoteSsh -Server $Server -Command $remoteCommand -SshExe $SshExe -BashExe $BashExe
    }
}


function Resolve-RemoteAppRoot {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ServerName,

        [string]$SshExe = "ssh",
        [string]$BashExe = ""
    )

    $probe = @'
set -e
for candidate in "$PWD" "$HOME/yolostudio_agent_proto" "/opt/yolostudio-agent"; do
  if [ -d "$candidate/agent_plan/agent" ]; then
    printf '%s\n' "$candidate"
    exit 0
  fi
done
printf '%s\n' "$HOME/yolostudio_agent_proto"
'@

    $result = Invoke-RemoteSshCapture -Server $ServerName -Command $probe -SshExe $SshExe -BashExe $BashExe
    if ($null -eq $result) {
        throw "Remote app root probe returned no result"
    }
    $resolved = (@($result) | Select-Object -Last 1).Trim()
    if ([string]::IsNullOrWhiteSpace($resolved)) {
        throw "Remote app root probe returned no result"
    }
    return $resolved
}


function Resolve-RemoteDatasetRoot {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ServerName,

        [string]$SshExe = "ssh",
        [string]$BashExe = ""
    )

    $probe = @'
set -e
for candidate in "$HOME/agent_cap_tests/zyb" "/data/example_dataset"; do
  if [ -d "$candidate" ]; then
    printf '%s\n' "$candidate"
    exit 0
  fi
done
printf '%s\n' "/data/example_dataset"
'@

    $result = Invoke-RemoteSshCapture -Server $ServerName -Command $probe -SshExe $SshExe -BashExe $BashExe
    if ($null -eq $result) {
        throw "Remote dataset root probe returned no result"
    }
    $resolved = (@($result) | Select-Object -Last 1).Trim()
    if ([string]::IsNullOrWhiteSpace($resolved)) {
        throw "Remote dataset root probe returned no result"
    }
    return $resolved
}


function Resolve-RemoteModelPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ServerName,

        [string]$SshExe = "ssh",
        [string]$BashExe = ""
    )

    $probe = @'
set -e
for candidate in "$HOME/yolov8n.pt" "/models/yolov8n.pt"; do
  if [ -f "$candidate" ]; then
    printf '%s\n' "$candidate"
    exit 0
  fi
done
printf '%s\n' "/models/yolov8n.pt"
'@

    $result = Invoke-RemoteSshCapture -Server $ServerName -Command $probe -SshExe $SshExe -BashExe $BashExe
    if ($null -eq $result) {
        throw "Remote model path probe returned no result"
    }
    $resolved = (@($result) | Select-Object -Last 1).Trim()
    if ([string]::IsNullOrWhiteSpace($resolved)) {
        throw "Remote model path probe returned no result"
    }
    return $resolved
}


function Sync-RemoteFiles {
    param(
        [Parameter(Mandatory = $true)]
        [hashtable[]]$Items,

        [string]$ScpExe = "scp",
        [string]$BashExe = ""
    )

    foreach ($item in $Items) {
        $args = @("-o", "BatchMode=yes", "-o", "ConnectTimeout=10")
        if ($item.ContainsKey('Recursive') -and $item.Recursive) {
            $args += "-r"
        }
        $args += @($item.Local, $item.Remote)
        Invoke-NativeChecked -Exe $ScpExe -Args $args -BashExe $BashExe
    }
}


function Sync-RemoteTextFileNormalized {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Server,

        [Parameter(Mandatory = $true)]
        [string]$LocalPath,

        [Parameter(Mandatory = $true)]
        [string]$RemotePath,

        [string]$SshExe = "ssh",
        [string]$ScpExe = "scp",
        [string]$BashExe = ""
    )

    Sync-RemoteFiles -Items @(
        @{
            Local = $LocalPath
            Remote = "$Server`:$RemotePath"
        }
    ) -ScpExe $ScpExe -BashExe $BashExe

    $remotePathLiteral = ConvertTo-BashLiteral -Value $RemotePath
    $normalizeCommand = "sed -i 's/\r`$//' $remotePathLiteral"
    Invoke-RemoteSsh -Server $Server -SshExe $SshExe -BashExe $BashExe -Command ($normalizeCommand.Trim())
}


function Fetch-RemoteFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Server,

        [Parameter(Mandatory = $true)]
        [string]$RemotePath,

        [Parameter(Mandatory = $true)]
        [string]$LocalPath,

        [string]$ScpExe = "scp",
        [string]$BashExe = ""
    )

    $localDir = Split-Path -Parent $LocalPath
    if (-not [string]::IsNullOrWhiteSpace($localDir)) {
        New-Item -ItemType Directory -Force -Path $localDir | Out-Null
    }

    Sync-RemoteFiles -Items @(
        @{
            Local = "$Server`:$RemotePath"
            Remote = $LocalPath
        }
    ) -ScpExe $ScpExe -BashExe $BashExe
}


function Sync-ManagedServerProtoMirror {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Server,

        [Parameter(Mandatory = $true)]
        [string]$RemoteAppRoot,

        [string]$SshExe = "ssh",
        [string]$ScpExe = "scp",
        [string]$BashExe = ""
    )

    $syncScript = Join-Path $PSScriptRoot "sync_server_proto_remote.ps1"
    $params = @{
        Server = $Server
        RemoteAppRoot = $RemoteAppRoot
        SshExe = $SshExe
        ScpExe = $ScpExe
    }
    if (-not [string]::IsNullOrWhiteSpace($BashExe)) {
        $params.BashExe = $BashExe
    }

    & $syncScript @params
    if ($LASTEXITCODE -ne 0) {
        throw "Remote server_proto mirror sync failed, exit code=$LASTEXITCODE"
    }
}
