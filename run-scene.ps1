[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Inputs,
    [string]$OutputDirectory,
    [string[]]$Stations,
    [switch]$Resume,
    [switch]$Check,
    [switch]$DepthWarp,
    [string]$SettingsPath = "$PSScriptRoot/scene_fusion/settings.local.json"
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $SettingsPath)) {
    throw 'Create scene_fusion/settings.local.json from settings.example.json and configure the Python environments and serving root.'
}
$settings = Get-Content -LiteralPath $SettingsPath -Raw | ConvertFrom-Json
$sceneArguments = @('scene_fusion/pipeline.py','--inputs',$Inputs,'--gpu-python',$settings.gpu_python)
if ($settings.serve_root) { $sceneArguments += @('--serve-root',$settings.serve_root) }
if ($OutputDirectory) { $sceneArguments += @('--out',$OutputDirectory) }
if ($Stations) { $sceneArguments += @('--stations') + $Stations }
if ($Resume) { $sceneArguments += '--resume' }
if ($Check) { $sceneArguments += '--check' }
if ($DepthWarp) { $sceneArguments += '--depth-warp' }
$env:OPENBLAS_NUM_THREADS = '1'
Push-Location $PSScriptRoot
try { & $settings.cpu_python @sceneArguments; $sceneExitCode = $LASTEXITCODE }
finally { Pop-Location }
exit $sceneExitCode
