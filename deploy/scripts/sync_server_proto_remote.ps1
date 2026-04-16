param(
    [string]$Server = "remote-agent",
    [string]$RemoteAppRoot = "",
    [string]$BashExe = "",
    [string]$SshExe = "",
    [string]$ScpExe = ""
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
$RepoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
. (Join-Path $PSScriptRoot "remote_script_common.ps1")
$LocalPython = Join-Path $RepoRoot "agent\.venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $LocalPython)) {
    $LocalPython = "python"
}

$ResolvedBashExe = $null
if (-not [string]::IsNullOrWhiteSpace($BashExe)) {
    $ResolvedBashExe = Resolve-NativeExecutable -Name 'bash' -ExplicitPath $BashExe
}
$ResolvedSshExe = if ([string]::IsNullOrWhiteSpace($SshExe)) { "ssh" } else { Resolve-NativeExecutable -Name 'ssh' -ExplicitPath $SshExe }
$ResolvedScpExe = if ([string]::IsNullOrWhiteSpace($ScpExe)) { "scp" } else { Resolve-NativeExecutable -Name 'scp' -ExplicitPath $ScpExe }

if ([string]::IsNullOrWhiteSpace($RemoteAppRoot)) {
    $RemoteAppRoot = Resolve-RemoteAppRoot -ServerName $Server -SshExe $ResolvedSshExe -BashExe $ResolvedBashExe
    Write-Host "resolved remote app root: $RemoteAppRoot"
}

Write-Host "==> refresh local server_proto mirror"
Invoke-NativeChecked -Exe $LocalPython -Args @((Join-Path $RepoRoot "deploy\scripts\sync_server_proto.py")) -BashExe $ResolvedBashExe

Write-Host "==> ensure remote mirror directories"
Ensure-RemoteDirectories -Server $Server -SshExe $ResolvedSshExe -BashExe $ResolvedBashExe -Commands @(
    "mkdir -p $RemoteAppRoot/agent_plan/agent && echo __REMOTE_READY__ agent",
    "mkdir -p $RemoteAppRoot/agent_plan/agent/client && echo __REMOTE_READY__ client",
    "mkdir -p $RemoteAppRoot/agent_plan/agent/server && echo __REMOTE_READY__ server",
    "mkdir -p $RemoteAppRoot/agent_plan/agent/tests && echo __REMOTE_READY__ tests",
    "mkdir -p $RemoteAppRoot/agent_plan/knowledge && echo __REMOTE_READY__ knowledge",
    "mkdir -p $RemoteAppRoot/agent_plan/yolostudio_agent && echo __REMOTE_READY__ namespace"
)

Write-Host "==> sync managed server_proto mirror"
Sync-RemoteFiles -ScpExe $ResolvedScpExe -BashExe $ResolvedBashExe -Items @(
    @{ Local = (Join-Path $RepoRoot "deploy\server_proto\agent_plan\__init__.py"); Remote = "$Server`:$RemoteAppRoot/agent_plan/__init__.py"; Recursive = $false },
    @{ Local = (Join-Path $RepoRoot "deploy\server_proto\agent_plan\agent\__init__.py"); Remote = "$Server`:$RemoteAppRoot/agent_plan/agent/__init__.py"; Recursive = $false },
    @{ Local = (Join-Path $RepoRoot "deploy\server_proto\agent_plan\agent\AGENT.md"); Remote = "$Server`:$RemoteAppRoot/agent_plan/agent/AGENT.md"; Recursive = $false },
    @{ Local = (Join-Path $RepoRoot "deploy\server_proto\agent_plan\agent\client"); Remote = "$Server`:$RemoteAppRoot/agent_plan/agent"; Recursive = $true },
    @{ Local = (Join-Path $RepoRoot "deploy\server_proto\agent_plan\agent\server"); Remote = "$Server`:$RemoteAppRoot/agent_plan/agent"; Recursive = $true },
    @{ Local = (Join-Path $RepoRoot "deploy\server_proto\agent_plan\agent\tests"); Remote = "$Server`:$RemoteAppRoot/agent_plan/agent"; Recursive = $true },
    @{ Local = (Join-Path $RepoRoot "deploy\server_proto\agent_plan\knowledge"); Remote = "$Server`:$RemoteAppRoot/agent_plan"; Recursive = $true },
    @{ Local = (Join-Path $RepoRoot "deploy\server_proto\agent_plan\yolostudio_agent"); Remote = "$Server`:$RemoteAppRoot/agent_plan"; Recursive = $true }
)

Write-Host "remote server_proto mirror sync finished"
Write-Host "remote app root: $RemoteAppRoot"
