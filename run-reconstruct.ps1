[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Inputs,
    [Parameter(Mandatory=$true)][string]$OutputDirectory,
    [ValidateSet('both','adaptive','smooth')][string]$Mode='both',
    [ValidateSet('standard','smoke')][string]$Profile='standard',
    [switch]$Resume,[switch]$Check,[switch]$Plan,
    [string]$SettingsPath="$PSScriptRoot/scene_fusion/settings.local.json"
)
$ErrorActionPreference='Stop'
$settings=Get-Content -LiteralPath $SettingsPath -Raw | ConvertFrom-Json
$arguments=@('scene_fusion/reconstruct.py','--inputs',$Inputs,'--out',$OutputDirectory,'--mode',$Mode,'--profile',$Profile,'--settings',$SettingsPath)
if($Resume){$arguments+='--resume'}
if($Check){$arguments+='--check'}
if($Plan){$arguments+='--plan'}
$env:OPENBLAS_NUM_THREADS='1'
Push-Location $PSScriptRoot
try { & $settings.cpu_python @arguments; $result=$LASTEXITCODE }
finally { Pop-Location }
exit $result
