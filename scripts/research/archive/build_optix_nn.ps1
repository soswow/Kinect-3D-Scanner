param(
    [string]$CudaRoot = 'F:/CUDA/12.4',
    [string]$VcVars = 'C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Auxiliary/Build/vcvars64.bat',
    [switch]$FetchHeaders
)
$ErrorActionPreference = 'Stop'
$taskRoot = (Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)))
$taskOutput = Join-Path $taskRoot 'benchmark-output/cuda-pipeline/optix-nearest'
$taskHeaders = Join-Path $taskOutput 'vendor/optix-dev-9.0.0/include'
if ($FetchHeaders) {
    $taskVendor = Join-Path $taskOutput 'vendor'
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
New-Item -ItemType Directory -Force -Path $taskOutput | Out-Null
$taskPtx = Join-Path $taskOutput 'nearest.ptx'
$taskDeviceSource = Join-Path $PSScriptRoot 'research_optix_nn.cu'
$taskSource = Join-Path $PSScriptRoot 'research_optix_nn_host.cpp'
$taskCommand = "call `"$VcVars`" >nul && `"$CudaRoot/bin/nvcc.exe`" --ptx --gpu-architecture=compute_86 --std=c++14 --fmad=false -I`"$taskHeaders`" `"$taskDeviceSource`" -o `"$taskPtx`" && cl /nologo /EHsc /std:c++17 /O2 /LD /I`"$taskHeaders`" /I`"$CudaRoot/include`" `"$taskSource`" /link /LIBPATH:`"$CudaRoot/lib/x64`" cuda.lib advapi32.lib /OUT:`"$taskOutput/nearest.dll`""
Push-Location $taskOutput
try {
    & $env:ComSpec /d /s /c $taskCommand
    if ($LASTEXITCODE -ne 0) { throw 'Standalone OptiX host bridge compilation failed.' }
} finally { Pop-Location }
Write-Output 'Offline host DLL/PTX compilation complete. No GPU context, scene or query was created.'
Get-FileHash -LiteralPath $taskPtx, (Join-Path $taskOutput 'nearest.dll')
