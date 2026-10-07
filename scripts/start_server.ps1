param([switch]$Background)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Project environment missing. Follow the Windows CUDA server setup in README.md.'
}

# Explicit requests make missing CUDA/native components fail visibly at startup.
$defaults = @{
    KINECT_DEVICE = 'cuda'
    KINECT_TRACKING = 'tensor'
    KINECT_NATIVE = 'on'
    KINECT_BLOCK_COUNT = '5000'
    KINECT_MAX_FRAMES = '500'
    KINECT_SERVER_HOST = '0.0.0.0'
    KINECT_SERVER_PORT = '8000'
    OMP_NUM_THREADS = '8'
    PYTHONUNBUFFERED = '1'
}
foreach ($entry in $defaults.GetEnumerator()) {
    if (-not [Environment]::GetEnvironmentVariable($entry.Key, 'Process')) {
        [Environment]::SetEnvironmentVariable($entry.Key, $entry.Value, 'Process')
    }
}

$port = [int]$env:KINECT_SERVER_PORT
if ($port -lt 1 -or $port -gt 65535) { throw 'KINECT_SERVER_PORT must be between 1 and 65535.' }
if (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue) {
    throw "TCP port $port is already in use. Stop the existing server or set KINECT_SERVER_PORT to another port."
}

if ($Background) {
    $logDir = Join-Path $projectRoot 'logs'
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    $logBase = Join-Path $logDir "server-$port"
    $process = Start-Process -FilePath $pythonPath -ArgumentList '-m', 'scanner_server' `
        -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput "$logBase.stdout.log" `
        -RedirectStandardError "$logBase.stderr.log"
    $healthHost = switch ($env:KINECT_SERVER_HOST) {
        '0.0.0.0' { '127.0.0.1' }
        '::' { '[::1]' }
        default { $env:KINECT_SERVER_HOST }
    }
    $healthUrl = "http://${healthHost}:$port/api/health"
    $deadline = (Get-Date).AddSeconds(180)
    try {
        while ($true) {
            if ($process.HasExited) {
                throw "Server exited with code $($process.ExitCode). See $logBase.stderr.log"
            }
            try {
                $health = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 2
                if ($health.status -eq 'ok') { break }
            } catch {
                # Retry while the server imports dependencies and initializes CUDA.
            }
            if ((Get-Date) -ge $deadline) {
                throw "Server startup timed out. See $logBase.stderr.log"
            }
            Start-Sleep -Milliseconds 250
        }
    } catch {
        if (-not $process.HasExited) {
            & taskkill /PID $process.Id /T /F | Out-Null
        }
        throw
    }
    Set-Content -LiteralPath "$logBase.pid" -Value $process.Id
    Write-Host "Server ready (PID $($process.Id)), device $($health.backend.device). Logs: $logBase.*.log"
    Write-Host "Health: $healthUrl"
    Write-Host "To stop this process and its Python child: taskkill /PID $($process.Id) /T /F"
} else {
    Push-Location $projectRoot
    try {
        Write-Host "Starting CUDA server on $($env:KINECT_SERVER_HOST):$env:KINECT_SERVER_PORT. Press Ctrl+C to stop."
        & $pythonPath -m scanner_server
        if ($LASTEXITCODE -ne 0) { throw "Server exited with code $LASTEXITCODE" }
    } finally {
        Pop-Location
    }
}
