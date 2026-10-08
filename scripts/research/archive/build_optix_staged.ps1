param(
    [string]$CudaRoot = 'F:/CUDA/12.4',
    [string]$VcVars = 'C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Auxiliary/Build/vcvars64.bat',
    [switch]$FetchHeaders
)
$ErrorActionPreference = 'Stop'
$taskRoot = (Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)))
$taskDependencyRoot = Join-Path $taskRoot 'benchmark-output/cuda-pipeline/optix-nearest'
$taskOutput = Join-Path $taskDependencyRoot 'staged-v1'
$taskHeaders = Join-Path $taskDependencyRoot 'vendor/optix-dev-9.0.0/include'
if ($FetchHeaders) {
    $taskVendor = Join-Path $taskDependencyRoot 'vendor'
    New-Item -ItemType Directory -Force -Path $taskVendor | Out-Null
    $taskHeaderZip = Join-Path $taskVendor 'optix-dev-v9.0.0.zip'
    if (-not (Test-Path -LiteralPath $taskHeaderZip)) {
        Invoke-WebRequest -Uri 'https://codeload.github.com/NVIDIA/optix-dev/zip/refs/tags/v9.0.0' -OutFile $taskHeaderZip
    }
    $taskExpectedZipHash = '461664fda99a901d364996bf048a3dae399ce1162c725e5c1855c336c6c9d0a8'
    if ((Get-FileHash -LiteralPath $taskHeaderZip -Algorithm SHA256).Hash.ToLowerInvariant() -ne $taskExpectedZipHash) {
        throw 'Official fixed-tag header ZIP differs from the recorded experiment dependency.'
    }
    Expand-Archive -LiteralPath $taskHeaderZip -DestinationPath $taskVendor -Force
    if (-not (Test-Path -LiteralPath (Join-Path $taskVendor 'optix-dev-9.0.0/LICENSE.txt'))) {
        throw 'The original NVIDIA SDK license must accompany the research headers.'
    }
}
if (-not (Test-Path -LiteralPath (Join-Path $taskHeaders 'optix.h'))) {
    throw 'Run with -FetchHeaders to retrieve official fixed-tag NVIDIA optix-dev v9.0.0 headers and retain their SDK license.'
}
# Bind the actual compiler headers to the same fixed archive as the report.
$taskHeaderZip = Join-Path $taskDependencyRoot 'vendor/optix-dev-v9.0.0.zip'
$taskExpectedZipHash = '461664fda99a901d364996bf048a3dae399ce1162c725e5c1855c336c6c9d0a8'
if ((Get-FileHash -LiteralPath $taskHeaderZip -Algorithm SHA256).Hash.ToLowerInvariant() -ne $taskExpectedZipHash) {
    throw 'Official header archive differs from the recorded fixed dependency.'
}
Add-Type -AssemblyName System.IO.Compression.FileSystem
$taskArchive = [IO.Compression.ZipFile]::OpenRead($taskHeaderZip)
try {
    foreach ($taskEntry in $taskArchive.Entries) {
        if ($taskEntry.FullName -notlike 'optix-dev-9.0.0/include/*' -or -not $taskEntry.Name) { continue }
        $taskLocalHeader = Join-Path (Join-Path $taskDependencyRoot 'vendor') $taskEntry.FullName
        if (-not (Test-Path -LiteralPath $taskLocalHeader)) { throw 'Required official compiler header is missing.' }
        $taskStream = $taskEntry.Open()
        $taskSha = [Security.Cryptography.SHA256]::Create()
        try { $taskArchiveHash = [BitConverter]::ToString($taskSha.ComputeHash($taskStream)).Replace('-', '').ToLowerInvariant() }
        finally { $taskStream.Dispose(); $taskSha.Dispose() }
        if ((Get-FileHash -LiteralPath $taskLocalHeader -Algorithm SHA256).Hash.ToLowerInvariant() -ne $taskArchiveHash) {
            throw "Extracted compiler header differs from its official archive: $($taskEntry.FullName)"
        }
    }
} finally { $taskArchive.Dispose() }
New-Item -ItemType Directory -Force -Path $taskOutput | Out-Null
$taskPtx = Join-Path $taskOutput 'nearest.ptx'
$taskDeviceSource = Join-Path $PSScriptRoot 'research_optix_staged.cu'
$taskSource = Join-Path $PSScriptRoot 'research_optix_staged_host.cpp'
$taskCommand = "call `"$VcVars`" >nul && `"$CudaRoot/bin/nvcc.exe`" --ptx --gpu-architecture=compute_86 --std=c++14 --fmad=false -I`"$taskHeaders`" `"$taskDeviceSource`" -o `"$taskPtx`" && cl /nologo /EHsc /std:c++17 /O2 /LD /I`"$taskHeaders`" /I`"$CudaRoot/include`" `"$taskSource`" /link /LIBPATH:`"$CudaRoot/lib/x64`" cuda.lib advapi32.lib /OUT:`"$taskOutput/nearest.dll`""
Push-Location $taskOutput
try {
    & $env:ComSpec /d /s /c $taskCommand
    if ($LASTEXITCODE -ne 0) { throw 'Standalone OptiX host bridge compilation failed.' }
} finally { Pop-Location }
Write-Output 'Offline host DLL/PTX compilation complete. No GPU context, scene or query was created.'
Get-FileHash -LiteralPath $taskPtx, (Join-Path $taskOutput 'nearest.dll')
