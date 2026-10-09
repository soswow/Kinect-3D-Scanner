<# Convenience wrapper for a source checkout. Startup policy lives in Python. #>
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Project environment missing. Follow Server Setup in README.md.'
}
Push-Location $projectRoot
try {
    & $pythonPath -m scanner_server @args
    $serverExit = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $serverExit
