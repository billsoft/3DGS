[CmdletBinding(PositionalBinding=$false)]
param(
    [string]$Python = 'C:/Users/bills/.codex/gsplat-fit-env/Scripts/python.exe',
    [string]$Cuda = 'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8',
    [string]$VisualStudio = 'C:/Program Files/Microsoft Visual Studio/18/Professional',
    [string]$Architecture = '8.9',
    [Parameter(ValueFromRemainingArguments=$true)][string[]]$PythonArguments
)
$ErrorActionPreference = 'Stop'
Import-Module (Join-Path $VisualStudio 'Common7/Tools/Microsoft.VisualStudio.DevShell.dll')
Enter-VsDevShell -VsInstallPath $VisualStudio -SkipAutomaticLocation -DevCmdArguments '-arch=x64 -host_arch=x64 -vcvars_ver=14.44' | Out-Null
$env:CUDA_HOME = $Cuda
$env:CUDA_PATH = $Cuda
$env:PATH = "$Cuda/bin;$(Split-Path $Python);$env:PATH"
$env:TORCH_CUDA_ARCH_LIST = $Architecture
$env:MAX_JOBS = '4'
$env:OPENBLAS_NUM_THREADS = '1'
& $Python @PythonArguments
exit $LASTEXITCODE
