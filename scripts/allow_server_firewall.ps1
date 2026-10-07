#Requires -RunAsAdministrator
param([int]$Port = 8000)

$ErrorActionPreference = 'Stop'
if ($Port -lt 1 -or $Port -gt 65535) { throw 'Port must be between 1 and 65535.' }
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Project environment missing.' }
$basePythonPath = & $pythonPath -c 'import sys; print(sys._base_executable)'
if ($LASTEXITCODE -ne 0) { throw 'Cannot determine the environment base interpreter.' }

# Windows venv launches a child using the base interpreter; cover both paths.
$programs = @($pythonPath, $basePythonPath) | Select-Object -Unique
$index = 0
foreach ($program in $programs) {
    $ruleName = "KinectScannerServer-$Port-$index"
    $existing = Get-NetFirewallRule -Name $ruleName -ErrorAction SilentlyContinue
    if ($existing) {
        Set-NetFirewallRule -Name $ruleName -Enabled True -Direction Inbound `
            -Action Allow -Protocol TCP -LocalPort $Port -RemoteAddress LocalSubnet `
            -Profile Any -Program $program | Out-Null
    } else {
        New-NetFirewallRule -Name $ruleName -DisplayName "Kinect Scanner Server (LAN, $Port, $index)" `
            -Direction Inbound -Action Allow -Protocol TCP -LocalPort $Port `
            -RemoteAddress LocalSubnet -Profile Any -Program $program | Out-Null
    }
    $index++
}
Write-Host "Allowed LAN connections to TCP $Port for the scanner's Python environment."
