param(
    [string]$RepoRoot = "D:\yolodo2.0",
    [string]$PromptPath = "D:\yolodo2.0\.codex\automation\mainline_autopilot_prompt.md",
    [string]$RunsRoot = "D:\yolodo2.0\.codex\automation\runs",
    [string]$CodexBin = "/mnt/c/Users/29615/.codex/bin/wsl/codex",
    [ValidateSet("danger", "full-auto")]
    [string]$ExecutionMode = "danger",
    [string]$Model = "",
    [string]$OperatorNote = "",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true

function Convert-ToWslPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$WindowsPath
    )

    $normalized = $WindowsPath -replace '\\', '/'
    if ($normalized -match '^([A-Za-z]):/(.*)$') {
        $drive = $matches[1].ToLowerInvariant()
        $rest = $matches[2]
        return "/mnt/$drive/$rest"
    }

    throw "无法转换为 WSL 路径: $WindowsPath"
}

function Write-Utf8NoBomFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,

        [Parameter(Mandatory = $true)]
        [string]$Content
    )

    $directory = Split-Path -Parent $Path
    if ($directory) {
        New-Item -ItemType Directory -Force -Path $directory | Out-Null
    }

    $encoding = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $Content, $encoding)
}

function Get-ShellQuoted {
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

    & $Exe @Args
    if ($LASTEXITCODE -ne 0) {
        throw "$Exe 执行失败，exit code=$LASTEXITCODE"
    }
}

if (!(Test-Path -LiteralPath $RepoRoot)) {
    throw "仓库路径不存在: $RepoRoot"
}

if (!(Test-Path -LiteralPath $PromptPath)) {
    throw "提示词文件不存在: $PromptPath"
}

$lockPath = Join-Path (Split-Path -Parent $RunsRoot) "autopilot.lock"
New-Item -ItemType Directory -Force -Path $RunsRoot | Out-Null

try {
    $lockStream = [System.IO.File]::Open($lockPath, [System.IO.FileMode]::OpenOrCreate, [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None)
}
catch {
    throw "自动推进已有实例在运行，锁文件: $lockPath"
}

try {
    $timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
    $runDir = Join-Path $RunsRoot $timestamp
    New-Item -ItemType Directory -Force -Path $runDir | Out-Null

    $promptText = [System.IO.File]::ReadAllText($PromptPath)
    $noteBlock = ""
    if ($OperatorNote.Trim()) {
        $noteBlock = "`n`n## 本轮额外要求`n`n$OperatorNote`n"
    }
    $runPrompt = $promptText + "`n`n## 本轮执行时间`n`n- " + (Get-Date).ToString("s") + $noteBlock

    $runPromptPath = Join-Path $runDir "prompt.md"
    Write-Utf8NoBomFile -Path $runPromptPath -Content $runPrompt

    $repoRootWsl = Convert-ToWslPath -WindowsPath $RepoRoot
    $runPromptWsl = Convert-ToWslPath -WindowsPath $runPromptPath
    $eventsPath = Join-Path $runDir "events.jsonl"
    $lastMessagePath = Join-Path $runDir "last_message.txt"
    $stderrPath = Join-Path $runDir "stderr.log"
    $metadataPath = Join-Path $runDir "metadata.json"
    $lastRunPath = Join-Path (Split-Path -Parent $RunsRoot) "last_run.json"

    $eventsPathWsl = Convert-ToWslPath -WindowsPath $eventsPath
    $lastMessagePathWsl = Convert-ToWslPath -WindowsPath $lastMessagePath
    $stderrPathWsl = Convert-ToWslPath -WindowsPath $stderrPath

    $execArgs = @("exec")
    if ($ExecutionMode -eq "danger") {
        $execArgs += "--dangerously-bypass-approvals-and-sandbox"
    }
    else {
        $execArgs += "--full-auto"
    }

    $execArgs += @(
        "-C", $repoRootWsl,
        "--json",
        "-o", $lastMessagePathWsl
    )

    if ($Model.Trim()) {
        $execArgs += @("-m", $Model)
    }

    $quotedExecArgs = $execArgs | ForEach-Object { Get-ShellQuoted -Value $_ }
    $codexCommand = @(
        "set -euo pipefail",
        "cd $(Get-ShellQuoted -Value $repoRootWsl)",
        "cat $(Get-ShellQuoted -Value $runPromptWsl) | $(Get-ShellQuoted -Value $CodexBin) $($quotedExecArgs -join ' ') > $(Get-ShellQuoted -Value $eventsPathWsl) 2> $(Get-ShellQuoted -Value $stderrPathWsl)"
    ) -join "; "

    $metadata = [ordered]@{
        started_at = (Get-Date).ToString("s")
        repo_root = $RepoRoot
        prompt_path = $runPromptPath
        execution_mode = $ExecutionMode
        model = $Model
        dry_run = [bool]$DryRun
        events_path = $eventsPath
        last_message_path = $lastMessagePath
        stderr_path = $stderrPath
    }

    if ($DryRun) {
        $metadata.status = "dry_run"
        $metadata.command = @("wsl.exe", "-e", "bash", "-lc", $codexCommand)
        $metadata.finished_at = (Get-Date).ToString("s")
        $metadata.exit_code = 0
        $metadataJson = $metadata | ConvertTo-Json -Depth 6
        Write-Utf8NoBomFile -Path $metadataPath -Content $metadataJson
        Write-Utf8NoBomFile -Path $lastRunPath -Content $metadataJson
        Write-Host "dry-run only"
        Write-Host "run dir: $runDir"
        Write-Host "command:"
        Write-Host $codexCommand
        return
    }

    try {
        Invoke-NativeChecked -Exe "wsl.exe" -Args @(
            "-e",
            "bash",
            "-lc",
            $codexCommand
        )

        $metadata.status = "completed"
        $metadata.finished_at = (Get-Date).ToString("s")
        $metadata.exit_code = 0
        $metadataJson = $metadata | ConvertTo-Json -Depth 6
        Write-Utf8NoBomFile -Path $metadataPath -Content $metadataJson
        Write-Utf8NoBomFile -Path $lastRunPath -Content $metadataJson

        Write-Host "codex mainline autopilot finished"
        Write-Host "run dir: $runDir"
    }
    catch {
        $metadata.status = "failed"
        $metadata.finished_at = (Get-Date).ToString("s")
        $metadata.exit_code = if ($LASTEXITCODE) { $LASTEXITCODE } else { 1 }
        $metadata.error = $_.Exception.Message
        $metadataJson = $metadata | ConvertTo-Json -Depth 6
        Write-Utf8NoBomFile -Path $metadataPath -Content $metadataJson
        Write-Utf8NoBomFile -Path $lastRunPath -Content $metadataJson
        throw
    }
}
finally {
    if ($lockStream) {
        $lockStream.Dispose()
    }
}
