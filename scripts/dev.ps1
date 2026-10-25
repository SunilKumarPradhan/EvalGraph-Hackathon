# Starts the EvalGraph backend (:8000) and frontend (:5173) in two new PowerShell windows.
# Usage:  .\scripts\dev.ps1
$root = Split-Path -Parent $PSScriptRoot
$backend = Join-Path $root "backend"
$frontend = Join-Path $root "frontend"

if (-not (Test-Path (Join-Path $backend ".env"))) {
    Copy-Item (Join-Path $backend ".env.example") (Join-Path $backend ".env")
    Write-Host "Created backend\.env from .env.example (mock mode). Edit it to use DIAL."
}
if (-not (Test-Path (Join-Path $frontend "node_modules"))) {
    Write-Host "Installing frontend dependencies..."
    Push-Location $frontend; npm install; Pop-Location
}

Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$backend'; `$env:VIRTUAL_ENV=`$null; uv run uvicorn evalgraph.api.main:app --reload --port 8000"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$frontend'; npm run dev"

Write-Host "API: http://localhost:8000/api/health"
Write-Host "UI:  http://localhost:5173   (add ?fixtures=1 to run the UI without the backend)"
