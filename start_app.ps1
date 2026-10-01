$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    throw "Project virtual environment not found at $python. Create it and install requirements first."
}

Get-Content .env | ForEach-Object {
    if ($_ -match '^\s*#' -or $_ -match '^\s*$') { return }
    $key, $value = $_ -split '=', 2
    if (-not [string]::IsNullOrWhiteSpace($key)) {
        [System.Environment]::SetEnvironmentVariable($key.Trim(), $value.Trim(), 'Process')
    }
}

@'
from backend.core.config import load_settings, validate_runtime_config
settings = load_settings()
validate_runtime_config(settings, mode='web', strict=False)
print('Runtime config check complete for web app')
'@ | & $python -

Write-Host 'Starting TruthLens web app...'
& $python -m backend.app
