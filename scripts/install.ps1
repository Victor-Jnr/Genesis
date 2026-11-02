$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    py -3 -m venv (Join-Path $root ".venv")
}
if (-not (Test-Path $python)) {
    Write-Error "Missing $python. Python 3.11+ is required to create the project virtual environment."
    exit 1
}

& $python -m pip install -r (Join-Path $root "requirements-dev.txt")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# Prebuilt wheels only. A source build needs a C++ toolchain and is not the project setup.
$indexes = @(
    "https://abetlen.github.io/llama-cpp-python/whl/cu128",
    "https://abetlen.github.io/llama-cpp-python/whl/cu124",
    "https://abetlen.github.io/llama-cpp-python/whl/cpu"
)
foreach ($index in $indexes) {
    & $python -m pip install --only-binary=:all: --extra-index-url $index "llama-cpp-python>=0.3.4"
    if ($LASTEXITCODE -ne 0) { continue }
    & $python -c "from llama_cpp import Llama"
    if ($LASTEXITCODE -eq 0) {
        Write-Output "Installed llama-cpp-python into .venv from $index"
        exit 0
    }
    & $python -m pip uninstall -y llama-cpp-python | Out-Null
}

Write-Error "No usable prebuilt llama-cpp-python wheel could be installed into .venv."
exit 1
