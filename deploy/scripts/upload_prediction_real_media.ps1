param(
    [Parameter(Mandatory = $true)]
    [string]$Server,

    [string]$StageDir = "D:\yolodo2.0\agent_plan\.tmp_prediction_real_media_stage",
    [string]$RemoteRoot = "/home/kly/prediction_real_media_stage"
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true

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

    Write-Host ("> " + $Exe + " " + ($Args -join " "))
    & $Exe @Args
    if ($LASTEXITCODE -ne 0) {
        throw "$Exe 执行失败，exit code=$LASTEXITCODE"
    }
}

Write-Host "==> ensure remote root"
$ensureRemoteRootCommand = "mkdir -p $RemoteRoot/weights $RemoteRoot/videos && echo __REMOTE_READY__"
Invoke-NativeChecked -Exe "ssh" -Args @(
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=10",
    $Server,
    $ensureRemoteRootCommand
)
Write-Host "==> remote root ready: $RemoteRoot"

Write-Host "==> upload manifest"
Invoke-NativeChecked -Exe "scp" -Args @(
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=10",
    $Manifest,
    "$Server`:$RemoteRoot/manifest.json"
)

Write-Host "==> upload weights"
Get-ChildItem -LiteralPath (Join-Path $StageDir "weights") -File | ForEach-Object {
    Invoke-NativeChecked -Exe "scp" -Args @(
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=10",
        $_.FullName,
        "$Server`:$RemoteRoot/weights/$($_.Name)"
    )
}

Write-Host "==> upload videos"
Get-ChildItem -LiteralPath (Join-Path $StageDir "videos") -File | ForEach-Object {
    Invoke-NativeChecked -Exe "scp" -Args @(
        "-o", "BatchMode=yes",
        "-o", "ConnectTimeout=10",
        $_.FullName,
        "$Server`:$RemoteRoot/videos/$($_.Name)"
    )
}

Write-Host "upload finished: $RemoteRoot"
