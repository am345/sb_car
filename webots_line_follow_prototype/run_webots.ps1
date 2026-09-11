$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$webotsCandidates = @(
  'C:\Program Files\Webots\msys64\mingw64\bin\webots.exe',
  'C:\Program Files\Webots\webots.exe'
)
$webotsExe = $webotsCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $webotsExe) {
  throw 'Webots R2025a is not installed. Run install_webots.ps1 first.'
}
$pythonDir = 'C:\Users\13567\.cache\codex-runtimes\codex-primary-runtime\dependencies\python'
$env:Path = "$pythonDir;$env:Path"
$env:PYTHONUTF8 = '1'
$env:WEBOTS_WEB_PORT = '9092'
& $webotsExe --batch --mode=fast (Join-Path $projectRoot 'worlds\line_follow.wbt')
