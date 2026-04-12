param(
    [string]$TaskName = "Codex-YoloDo-Mainline-Autopilot",
    [string]$RunnerPath = "D:\yolodo2.0\agent_plan\deploy\scripts\run_codex_mainline_autopilot.ps1",
    [int]$IntervalMinutes = 30,
    [ValidateSet("danger", "full-auto")]
    [string]$ExecutionMode = "danger",
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true

if ($IntervalMinutes -lt 5) {
    throw "IntervalMinutes 不能小于 5"
}

if (!(Test-Path -LiteralPath $RunnerPath)) {
    throw "runner 脚本不存在: $RunnerPath"
}

$startTime = (Get-Date).AddMinutes(1).ToString("HH:mm")
$runnerArgument = "-NoProfile -ExecutionPolicy Bypass -File `"$RunnerPath`" -ExecutionMode $ExecutionMode"
$taskCommand = "powershell.exe $runnerArgument"

$schtasksArgs = @(
    "/Create",
    "/TN", $TaskName,
    "/SC", "MINUTE",
    "/MO", "$IntervalMinutes",
    "/ST", $startTime,
    "/TR", $taskCommand,
    "/F"
)

if ($DryRun) {
    Write-Host "dry-run only"
    Write-Host ("> schtasks.exe " + ($schtasksArgs -join " "))
    return
}

& schtasks.exe @schtasksArgs
if ($LASTEXITCODE -ne 0) {
    throw "schtasks.exe 执行失败，exit code=$LASTEXITCODE"
}

Write-Host "scheduled task created"
Write-Host "task: $TaskName"
Write-Host "interval: every $IntervalMinutes minutes"
