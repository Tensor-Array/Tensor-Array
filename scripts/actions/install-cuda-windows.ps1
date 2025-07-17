$CUDA_PACKAGES_IN = @(
    "nvcc"
    "visual_studio_integration"
    "cudart"
    "nvtx"
    "nvrtc"
    "thrust"
    "curand_dev"
    "cublas_dev"
    "cufft_dev"
)

function Version-Ge($a, $b) {
    return ([version]$a -ge [version]$b)
}
function Version-Gt($a, $b) {
    return ([version]$a -gt [version]$b)
}
function Version-Le($a, $b) {
    return ([version]$a -le [version]$b)
}
function Version-Lt($a, $b) {
    return ([version]$a -lt [version]$b)
}

# Expect $env:cuda to be set, e.g. "12.4.1"
$CUDA_VERSION_MAJOR_MINOR = $env:cuda

$parts = $CUDA_VERSION_MAJOR_MINOR.Split('.')
$CUDA_MAJOR = $parts[0]
$CUDA_MINOR = $parts[1]
$CUDA_PATCH = if ($parts.Count -gt 2) { $parts[2] } else { "0" }

$CUDA_PACKAGES = ""
foreach ($package in $CUDA_PACKAGES_IN) {
    $CUDA_PACKAGES += " ${package}_$CUDA_MAJOR.$CUDA_MINOR"
}
Write-Host "CUDA_PACKAGES $CUDA_PACKAGES"

$cudaInstallerUrl = "https://developer.download.nvidia.com/compute/cuda/$CUDA_VERSION_MAJOR_MINOR/network_installers/cuda_${CUDA_VERSION_MAJOR_MINOR}_windows_network.exe"
Invoke-WebRequest -Uri $cudaInstallerUrl -OutFile "cuda_installer.exe"
Start-Process -FilePath ".\cuda_installer.exe" -ArgumentList "-s $CUDA_PACKAGES" -Wait
Remove-Item "cuda_installer.exe" -Force

$CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v$CUDA_MAJOR.$CUDA_MINOR"
Write-Host "CUDA_PATH=$CUDA_PATH"
$env:CUDA_PATH = $CUDA_PATH

# If executing on github actions, emit the appropriate echo statements to update environment variables
if (Test-Path "env:GITHUB_ACTIONS") {
    # Set paths for subsequent steps, using $env:CUDA_PATH
    Write-Host "Adding CUDA to CUDA_PATH, and PATH"
    Add-Content -Path $env:GITHUB_ENV -Value "CUDA_PATH=$env:CUDA_PATH"
    Add-Content -Path $env:GITHUB_PATH -Value "$env:CUDA_PATH\bin"
}
