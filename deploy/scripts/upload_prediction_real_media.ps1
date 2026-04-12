param(
    [Parameter(Mandatory = $true)]
    [string]$Server,

    [string]$StageDir = "",
    [string]$RemoteRoot = "/data/prediction_real_media_stage"
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
$RepoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
if ([string]::IsNullOrWhiteSpace($StageDir)) {
    $StageDir = Join-Path $RepoRoot ".tmp_prediction_real_media_stage"
}

if (!(Test-Path -LiteralPath $StageDir)) {
    throw "StageDir 不存在: $StageDir"
}

$Manifest = Join-Path $StageDir "manifest.json"
if (!(Test-Path -LiteralPath $Manifest)) {
    throw "manifest.json 不存在，请先运行 stage_prediction_real_media.py"
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

Write-Host "==> ensure remote root"
$ensureRootCommands = @(
    "mkdir -p $RemoteRoot && echo __REMOTE_READY__ root",
    "mkdir -p $RemoteRoot/weights && echo __REMOTE_READY__ weights",
    "mkdir -p $RemoteRoot/videos && echo __REMOTE_READY__ videos"
)
foreach ($remoteCommand in $ensureRootCommands) {
    Invoke-NativeChecked -Exe "ssh" -Args @("-n", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", $Server, $remoteCommand)
}
Write-Host "==> remote root ready: $RemoteRoot"

Write-Host "==> upload manifest"
Invoke-NativeChecked -Exe "scp" -Args @("-o", "BatchMode=yes", "-o", "ConnectTimeout=10", $Manifest, "$Server`:$RemoteRoot/manifest.json")

Write-Host "==> upload weights"
Get-ChildItem -LiteralPath (Join-Path $StageDir "weights") -File | ForEach-Object {
    Invoke-NativeChecked -Exe "scp" -Args @("-o", "BatchMode=yes", "-o", "ConnectTimeout=10", $_.FullName, "$Server`:$RemoteRoot/weights/$($_.Name)")
}

Write-Host "==> upload videos"
Get-ChildItem -LiteralPath (Join-Path $StageDir "videos") -File | ForEach-Object {
    Invoke-NativeChecked -Exe "scp" -Args @("-o", "BatchMode=yes", "-o", "ConnectTimeout=10", $_.FullName, "$Server`:$RemoteRoot/videos/$($_.Name)")
}

Write-Host "upload finished: $RemoteRoot"
