# Starts the EvalGraph backend (:8000) and frontend (:5173) in a given LLM mode, for manual UI testing.
# Usage:  .\run-evalgraph.ps1 <mock|real|replay>
#         .\run-evalgraph.ps1 -Stop        (closes both windows and their uvicorn/npm processes)
#   mock    - EVALGRAPH_MOCK_LLM=1, ARCHER_MOCK_LLM=1   : no DIAL calls, canned output, UI shows MOCK. No key/VPN needed.
#   real    - EVALGRAPH_MOCK_LLM=0, ARCHER_MOCK_LLM=0   : live DIAL calls. Needs DIAL_API_KEY in backend\.env and the VPN.
#   replay  - EVALGRAPH_REPLAY=1 (mock stays 1 as fallback) : serves the recorded fixtures/demo run, UI shows REPLAY.
# Mode is set as process env vars for the spawned backend window only; backend\.env is never edited.
param(
    [Parameter(Position = 0)]
    [ValidateSet("mock", "real", "replay")]
    [string]$Mode = "mock",
    [switch]$Stop
)
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$backend = Join-Path $root "backend"
$frontend = Join-Path $root "frontend"
$envFile = Join-Path $backend ".env"
$pidFile = Join-Path $root ".evalgraph-run.pids"

function Stop-Tree([int]$ProcId) {
    Get-CimInstance Win32_Process -Filter "ParentProcessId=$ProcId" -ErrorAction SilentlyContinue |
        ForEach-Object { Stop-Tree $_.ProcessId }
    Stop-Process -Id $ProcId -Force -ErrorAction SilentlyContinue
}

if ($Stop) {
    if (-not (Test-Path $pidFile)) {
        Write-Host "Nothing to stop (no $pidFile - was it started with this script?)."
        exit 0
    }
    $pids = Get-Content $pidFile | Where-Object { $_ -match '^\d+$' }
    foreach ($p in $pids) {
        Write-Host "Stopping process tree $p"
        Stop-Tree ([int]$p)
    }
    Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
    Write-Host "Stopped backend and frontend."
    exit 0
}

if (-not (Test-Path $envFile)) {
    Copy-Item (Join-Path $backend ".env.example") $envFile
    Write-Host "Created backend\.env from .env.example. Add DIAL_API_KEY there before using -Mode real."
}

switch ($Mode) {
    "mock" { $mockLlm = "1"; $replay = "0"; $archerMock = "1" }
    "real" { $mockLlm = "0"; $replay = "0"; $archerMock = "0" }
    "replay" { $mockLlm = "1"; $replay = "1"; $archerMock = "1" }
}

if ($Mode -eq "real") {
    $hasKey = (Get-Content $envFile | Where-Object { $_ -match "^DIAL_API_KEY=\S" })
    if (-not $hasKey) {
        Write-Host "DIAL_API_KEY is empty in backend\.env - real mode will fail to call DIAL." -ForegroundColor Yellow
    }
    Write-Host "real mode needs the company VPN (DIAL only works on-network)." -ForegroundColor Yellow
}

if (-not (Test-Path (Join-Path $frontend "node_modules"))) {
    Write-Host "Installing frontend dependencies..."
    Push-Location $frontend; npm install; Pop-Location
}

$backendCmd = "Set-Location '$backend'; " +
    "`$env:VIRTUAL_ENV=`$null; " +
    "`$env:EVALGRAPH_MOCK_LLM='$mockLlm'; " +
    "`$env:EVALGRAPH_REPLAY='$replay'; " +
    "`$env:ARCHER_MOCK_LLM='$archerMock'; " +
    "uv run --no-sync uvicorn evalgraph.api.main:app --reload --port 8000"

$backendProc = Start-Process powershell -ArgumentList "-NoExit", "-Command", $backendCmd -PassThru
$frontendProc = Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$frontend'; npm run dev" -PassThru

@($backendProc.Id, $frontendProc.Id) | Set-Content $pidFile

Write-Host ""
Write-Host "Mode: $Mode  (EVALGRAPH_MOCK_LLM=$mockLlm EVALGRAPH_REPLAY=$replay ARCHER_MOCK_LLM=$archerMock)"
Write-Host "API: http://127.0.0.1:8000/api/health"
Write-Host "UI:  http://localhost:5173"
Write-Host "Stop with: .\run-evalgraph.ps1 -Stop"
