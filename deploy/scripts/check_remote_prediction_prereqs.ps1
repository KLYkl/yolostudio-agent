param(
    [string]$ServerHost = '203.0.113.10',
    [string]$ServerAlias = 'yolostudio',
    [int[]]$Ports = @(22, 8080, 11434)
)

$ErrorActionPreference = 'Stop'
$env:SystemRoot = 'C:\Windows'
$env:WINDIR = 'C:\Windows'
$env:ComSpec = 'C:\Windows\System32\cmd.exe'

function Test-TcpPort {
    param(
        [string]$TargetHost,
        [int]$Port
    )

    try {
        $client = [System.Net.Sockets.TcpClient]::new()
        $iar = $client.BeginConnect($TargetHost, $Port, $null, $null)
        $ok = $iar.AsyncWaitHandle.WaitOne(3000, $false)
        if (-not $ok) {
            $client.Close()
            return [pscustomobject]@{ host = $TargetHost; port = $Port; ok = $false; detail = 'timeout' }
        }
        try {
            $client.EndConnect($iar)
            $client.Close()
            return [pscustomobject]@{ host = $TargetHost; port = $Port; ok = $true; detail = 'connected' }
        }
        catch {
            $client.Close()
            return [pscustomobject]@{ host = $TargetHost; port = $Port; ok = $false; detail = $_.Exception.InnerException.Message }
        }
    }
    catch {
        return [pscustomobject]@{ host = $TargetHost; port = $Port; ok = $false; detail = $_.Exception.Message }
    }
}

Write-Host "==> remote TCP preflight: $ServerHost"
$results = foreach ($port in $Ports) {
    Test-TcpPort -TargetHost $ServerHost -Port $port
}
$results | Format-Table -AutoSize

Write-Host "==> ssh executable"
$ssh = Get-Command ssh -ErrorAction SilentlyContinue
if ($null -eq $ssh) {
    Write-Host 'ssh: missing'
}
else {
    Write-Host ("ssh: " + $ssh.Source)
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $ssh.Source
    @('-o','BatchMode=yes','-o','ConnectTimeout=5',$ServerAlias,'echo codex-remote-ok') | ForEach-Object { [void]$psi.ArgumentList.Add($_) }
    $psi.Environment['SystemRoot'] = 'C:\Windows'
    $psi.Environment['WINDIR'] = 'C:\Windows'
    $psi.Environment['ComSpec'] = 'C:\Windows\System32\cmd.exe'
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $proc = [System.Diagnostics.Process]::Start($psi)
    $proc.WaitForExit()
    Write-Host ("ssh_exit=" + $proc.ExitCode)
    $stdout = $proc.StandardOutput.ReadToEnd().Trim()
    $stderr = $proc.StandardError.ReadToEnd().Trim()
    if ($stdout) { Write-Host ("ssh_stdout=" + $stdout) }
    if ($stderr) { Write-Host ("ssh_stderr=" + $stderr) }
}
