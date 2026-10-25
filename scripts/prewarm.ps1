# Run ~10 minutes before the demo. Checks everything that can fail live and warms caches.
# Usage:  .\scripts\prewarm.ps1 [-Api http://127.0.0.1:8000] [-RepoUrl https://github.com/SunilKumarPradhan/cuda-archer]
# Prints a checklist; exit code 1 if anything the live demo needs is broken (with the fallback to use).
param(
    [string]$Api = "http://127.0.0.1:8000",
    [string]$RepoUrl = "https://github.com/SunilKumarPradhan/cuda-archer"
)
$root = Split-Path -Parent $PSScriptRoot
$archer = Join-Path (Split-Path -Parent $root) "cuda-archer"
$env:VIRTUAL_ENV = $null
$failed = @()

function Check([string]$name, [scriptblock]$test, [string]$fallback) {
    $watch = [Diagnostics.Stopwatch]::StartNew()
    try {
        $detail = & $test
        Write-Host ("  OK    {0}  ({1:N1}s) {2}" -f $name, $watch.Elapsed.TotalSeconds, $detail) -ForegroundColor Green
    }
    catch {
        Write-Host ("  FAIL  {0}: {1}" -f $name, $_.Exception.Message) -ForegroundColor Red
        if ($fallback) { Write-Host "        -> $fallback" -ForegroundColor Yellow }
        $script:failed += $name
    }
}

Write-Host "EvalGraph pre-warm"
Check "CUDA Archer models downloaded (embeddings, reranker)" {
    Push-Location $archer
    try { cmd /c "uv run --no-sync python -m cuda_archer.rag.ingest >nul 2>&1"; if ($LASTEXITCODE) { throw "ingest exited $LASTEXITCODE" } }
    finally { Pop-Location }
} "check internet access to Hugging Face; models are cached after one successful run"

Check "DIAL answers (VPN up, deployments exist)" {
    Push-Location $archer
    try {
        $out = (cmd /c "uv run --no-sync python -m cuda_archer.dial_check 2>&1") -join "`n"
        $last = ($out.Trim() -split "`n")[-1].Trim()
        if ($LASTEXITCODE) { throw $last }
        $last
    }
    finally { Pop-Location }
} "connect VPN, or start the backend with EVALGRAPH_REPLAY=1 and present the recorded run"

Check "backend health" {
    $h = Invoke-RestMethod "$Api/api/health" -TimeoutSec 5
    if ($h.mock) { "MOCK mode (EVALGRAPH_MOCK_LLM=1): scores will be pseudo values" }
    elseif ($h.replay) { "REPLAY mode" }
    else { "DIAL $($h.model), judge $($h.judge_model)" }
} "start it: .\scripts\dev.ps1"

Check "replay recording present" {
    $m = Join-Path $root "fixtures\demo\manifest.json"
    if (-not (Test-Path $m)) { throw "no fixtures\demo\manifest.json" }
    $j = Get-Content $m -Raw | ConvertFrom-Json
    if ($j.mock) { "recorded $($j.recorded_at) in MOCK mode: rehearsal only" } else { "recorded $($j.recorded_at), $($j.runs.Count) runs" }
} "record one on VPN: .\scripts\record_demo.ps1"

Check "demo repo reachable on GitHub" {
    git ls-remote --heads $RepoUrl 2>&1 | Out-Null
    if ($LASTEXITCODE) { throw "git ls-remote failed (private repo without credentials, or no network)" }
} "USE LOCAL PATH FALLBACK: Home -> 'Use a local folder instead' -> ..\..\cuda-archer"

Check "docker running (sandbox runs of other repos)" {
    # exits 0 even when Docker Desktop says "unable to start", so check for a real version string
    $version = (docker info --format "{{.ServerVersion}}" 2>&1 | Out-String).Trim()
    if ($LASTEXITCODE -or -not $version -or $version -match "error") { throw "docker daemon not reachable: $version" }
    "server $version"
} "start Docker Desktop, or keep the 'judge's repo' moment analysis-only"

Check "end-to-end smoke (2 samples, warms connections + LLM cache)" {
    $log = Join-Path $env:TEMP "evalgraph-prewarm-smoke.log"
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "smoke.ps1") -Limit 2 -Api $Api *> $log
    if ($LASTEXITCODE) { throw "smoke.ps1 failed (log: $log)" }
    (Get-Content $log | Select-String "SMOKE PASSED|compare baseline").Line -join " | "
} "run .\scripts\smoke.ps1 -Limit 2 to see the error"

if ($failed.Count) { Write-Host "`n$($failed.Count) check(s) failed: $($failed -join ', ')" -ForegroundColor Red; exit 1 }
Write-Host "`nREADY" -ForegroundColor Green
