$ErrorActionPreference = 'Stop'
$installer = Join-Path (Split-Path -Parent $PSScriptRoot) 'tmp\webots-R2025a_setup.exe'
if (-not (Test-Path -LiteralPath $installer)) {
  throw "Installer not found: $installer"
}
Start-Process -FilePath $installer -ArgumentList '/VERYSILENT','/SUPPRESSMSGBOXES','/NORESTART','/SP-' -Wait -WindowStyle Hidden
Write-Host 'Webots R2025a installation completed.'
