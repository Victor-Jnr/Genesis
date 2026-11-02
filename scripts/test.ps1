$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    Write-Error "Missing $python. Create the project environment with: py -3 -m venv .venv"
    exit 1
}
& $python -m pytest @args
exit $LASTEXITCODE
