<# Start a CUDA scanner server using a reproducible measured recipe. #>
[CmdletBinding()]
param(
    [string]$Python,
    [ValidateSet('baseline', 'hybrid', 'adaptive')][string]$Recipe = 'hybrid',
    [ValidateSet('off', 'auto', 'on')][string]$CudaInput = 'off',
    [ValidateSet('off', 'auto', 'on')][string]$CudaConfidence = 'off',
    [ValidateRange(1024, 65535)][int]$Port = 8000,
    [string]$BindAddress = '0.0.0.0'
)

$ErrorActionPreference = 'Stop'

function Get-ProcessIdentity([int]$ProcessId) {
    $process = Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId" -ErrorAction SilentlyContinue
    if ($null -eq $process) { return $null }
    [pscustomobject]@{
        Id = [int]$process.ProcessId
        ParentId = [int]$process.ParentProcessId
        CreatedTicks = ([DateTime]$process.CreationDate).ToUniversalTime().Ticks
    }
}

function Get-OwnedProcessChain([int]$ProcessId, $RootIdentity) {
    $chain = @()
    $seen = @{}
    $child = $null
    while ($ProcessId -gt 0 -and -not $seen.ContainsKey($ProcessId)) {
        $seen[$ProcessId] = $true
        $identity = Get-ProcessIdentity $ProcessId
        if ($null -eq $identity -or $identity.CreatedTicks -lt $RootIdentity.CreatedTicks) { return }
        # A parent created after its child indicates a recycled parent PID.
        if ($null -ne $child -and $identity.CreatedTicks -gt $child.CreatedTicks) { return }
        $chain += $identity
        if ($identity.Id -eq $RootIdentity.Id) {
            if ($identity.CreatedTicks -eq $RootIdentity.CreatedTicks) { return $chain }
            return
        }
        $child = $identity
        $ProcessId = $identity.ParentId
    }
}

function Test-OwnedListener([int]$ListenerPort, $RootIdentity, $KnownProcesses) {
    $listeners = @(Get-NetTCPConnection -LocalPort $ListenerPort -State Listen -ErrorAction SilentlyContinue)
    if ($listeners.Count -eq 0) { return $false }
    foreach ($listener in $listeners) {
        $chain = @(Get-OwnedProcessChain ([int]$listener.OwningProcess) $RootIdentity)
        if ($chain.Count -eq 0) {
            throw "Port $ListenerPort is owned by an unrelated process; refusing its health response."
        }
        foreach ($identity in $chain) { $KnownProcesses[$identity.Id] = $identity }
    }
    return $true
}

function Stop-OwnedLaunch($RootIdentity, $KnownProcesses) {
    # Capture descendants while the original creation identities can still be
    # verified. Retain listener descendants already observed if the root exited.
    $snapshot = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue)
    $changed = $true
    while ($changed) {
        $changed = $false
        foreach ($process in $snapshot) {
            $processId = [int]$process.ProcessId
            $parentId = [int]$process.ParentProcessId
            if ($KnownProcesses.ContainsKey($processId) -or -not $KnownProcesses.ContainsKey($parentId)) { continue }
            $parent = $KnownProcesses[$parentId]
            $currentParent = $snapshot | Where-Object { [int]$_.ProcessId -eq $parentId } | Select-Object -First 1
            if ($null -eq $currentParent -or ([DateTime]$currentParent.CreationDate).ToUniversalTime().Ticks -ne $parent.CreatedTicks) { continue }
            $createdTicks = ([DateTime]$process.CreationDate).ToUniversalTime().Ticks
            if ($createdTicks -lt $parent.CreatedTicks) { continue }
            $KnownProcesses[$processId] = [pscustomobject]@{ Id = $processId; ParentId = $parentId; CreatedTicks = $createdTicks }
            $changed = $true
        }
    }
    $currentRoot = Get-ProcessIdentity $RootIdentity.Id
    if ($null -ne $currentRoot -and $currentRoot.CreatedTicks -eq $RootIdentity.CreatedTicks) {
        try {
            & "$env:SystemRoot/System32/taskkill.exe" /PID $RootIdentity.Id /T /F 2>$null | Out-Null
            $killExitCode = $LASTEXITCODE
            if ($killExitCode -ne 0) { Write-Warning "taskkill exited $killExitCode; checking owned process identities for cleanup." }
        } catch {
            Write-Warning "taskkill failed; checking owned process identities for cleanup: $_"
        }
    }
    foreach ($identity in ($KnownProcesses.Values | Sort-Object CreatedTicks -Descending)) {
        $current = Get-ProcessIdentity $identity.Id
        if ($null -eq $current -or $current.CreatedTicks -ne $identity.CreatedTicks) { continue }
        $handle = Get-Process -Id $identity.Id -ErrorAction SilentlyContinue
        if ($null -eq $handle) { continue }
        try {
            # Acquire this process handle before the final identity check so a
            # later PID reuse cannot redirect Kill() to an unrelated process.
            $null = $handle.Handle
            $current = Get-ProcessIdentity $identity.Id
            if ($null -ne $current -and $current.CreatedTicks -eq $identity.CreatedTicks -and -not $handle.HasExited) { $handle.Kill() }
        } catch { Write-Warning "Unable to stop owned process $($identity.Id): $_" }
        finally { $handle.Dispose() }
    }
}

$workspaceRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
if (-not $Python) {
    $pythonCandidates = @((Join-Path $workspaceRoot '.venv/Scripts/python.exe'))
    $commonGitDirectory = (& git -C $workspaceRoot rev-parse --path-format=absolute --git-common-dir).Trim()
    if ($LASTEXITCODE -eq 0) {
        $sharedCheckoutRoot = Split-Path -Path $commonGitDirectory -Parent
        $pythonCandidates += Join-Path $sharedCheckoutRoot '.venv/Scripts/python.exe'
    }
    $Python = $pythonCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if (-not $Python) { $Python = (Get-Command python -ErrorAction Stop).Source }
}
if (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue) {
    throw "Port $Port is already in use. Stop the existing server or select another port."
}

$env:OMP_NUM_THREADS = '8'
$env:KINECT_NATIVE = 'on'
$env:KINECT_DEVICE = 'cuda'
$env:KINECT_BLOCK_COUNT = '5000'
$env:KINECT_MAX_FRAMES = '500'
$env:KINECT_CUDA_FUSION = 'fused'
$env:KINECT_CUDA_REGISTRATION = 'cpu'
$env:KINECT_CUDA_ODOMETRY = 'off'
$env:KINECT_LIVE_RECOVERY = 'full'
$env:KINECT_VISUAL_FEATURES = 'orb'
$env:KINECT_ADAPTIVE_EXPERIMENTAL = 'off'
$env:KINECT_CUDA_INPUT = $CudaInput
$env:KINECT_CUDA_CONFIDENCE = $CudaConfidence
$env:KINECT_VISUAL_REFINEMENT = 'icp'
$env:KINECT_FINAL_VISUAL_FIRST = 'off'
$env:KINECT_FINAL_LOCAL_REFINEMENT = 'icp'
$env:KINECT_SERVER_HOST = $BindAddress
$env:KINECT_SERVER_PORT = [string]$Port
$env:PYTHONUNBUFFERED = '1'
if ($Recipe -in @('hybrid', 'adaptive')) {
    $env:KINECT_TRACKING = 'legacy'
    $env:KINECT_CUDA_MATCHING = 'cuda'
    $env:KINECT_KEYFRAME_CACHE = 'on'
    $env:KINECT_MODEL_REFRESH = 'lazy'
    if ($Recipe -eq 'adaptive') { $env:KINECT_VISUAL_FEATURES = 'adaptive' }
} else {
    $env:KINECT_TRACKING = 'tensor'
    $env:KINECT_CUDA_MATCHING = 'cpu'
    $env:KINECT_KEYFRAME_CACHE = 'off'
    $env:KINECT_MODEL_REFRESH = 'eager'
}
$logDirectory = Join-Path $workspaceRoot 'logs'
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
$pidPath = Join-Path $logDirectory 'server.pid'
$healthHost = switch ($BindAddress) {
    '0.0.0.0' { '127.0.0.1' }
    '::' { '[::1]' }
    default { if ($BindAddress.Contains(':')) { "[$BindAddress]" } else { $BindAddress } }
}
$serverProcess = $null
$rootIdentity = $null
$knownProcesses = @{}
$serverReady = $false
try {
    $serverProcess = Start-Process -FilePath $Python -ArgumentList '-m', 'scanner_server' `
        -WorkingDirectory $workspaceRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $logDirectory 'server.stdout.log') `
        -RedirectStandardError (Join-Path $logDirectory 'server.stderr.log')
    $rootIdentity = Get-ProcessIdentity $serverProcess.Id
    if ($serverProcess.HasExited -or $null -eq $rootIdentity) { throw "Server exited before its process identity could be recorded." }
    $knownProcesses[$rootIdentity.Id] = $rootIdentity
    Set-Content -LiteralPath $pidPath -Value $serverProcess.Id
    $readyDeadline = [DateTime]::UtcNow.AddMinutes(3)
    while ([DateTime]::UtcNow -lt $readyDeadline) {
        if ($serverProcess.HasExited) { throw "Server exited; see $logDirectory/server.stderr.log" }
        $ownedListener = Test-OwnedListener $Port $rootIdentity $knownProcesses
        try {
            $health = if ($ownedListener) { Invoke-RestMethod -Uri "http://${healthHost}:$Port/api/health" -TimeoutSec 2 } else { $null }
        } catch { $health = $null }
        if ($null -ne $health -and $health.status -eq 'ok') {
            if (-not (Test-OwnedListener $Port $rootIdentity $knownProcesses)) { continue }
            if ($health.backend.device -ne 'CUDA:0') { throw "Launched server did not select CUDA:0." }
            if ($health.backend.cuda_input.requested -ne $CudaInput -or $health.backend.depth_confidence.requested -ne $CudaConfidence) { throw "Launched server did not apply the requested CUDA input/confidence flags." }
            $serverReady = $true
            Write-Output "CUDA server ready on port $Port; recipe=$Recipe; PID=$($serverProcess.Id)"
            if ($Recipe -eq 'adaptive') {
                Write-Output 'Adaptive recovery activates at Live voxel 10 mm / Final voxel 5 mm; other resolutions use ORB. Enable color-assisted recovery.'
            }
            return
        }
        Start-Sleep -Milliseconds 250
    }
    throw "Server health check timed out; see $logDirectory/server.stderr.log"
} finally {
    if (-not $serverReady -and $null -ne $serverProcess) {
        if ($null -ne $rootIdentity) { Stop-OwnedLaunch $rootIdentity $knownProcesses }
        elseif (-not $serverProcess.HasExited) { $serverProcess.Kill() }
        $recordedPid = Get-Content -LiteralPath $pidPath -Raw -ErrorAction SilentlyContinue
        if ($recordedPid -and $recordedPid.Trim() -eq [string]$serverProcess.Id) {
            Remove-Item -LiteralPath $pidPath -ErrorAction SilentlyContinue
        }
    }
}
