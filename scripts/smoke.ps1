# End-to-end smoke test through the HTTP API: analyze CUDA Archer -> run baseline -> run improved -> compare.
# Usage:  .\scripts\smoke.ps1 [-Limit 10] [-Api http://127.0.0.1:8000]
# Starts the backend itself if nothing answers on -Api, and stops it afterwards. Exit code 1 on any failure.
# In mock mode (backend\.env EVALGRAPH_MOCK_LLM=1) judge scores are pseudo values: the run proves plumbing, not quality.
# Use 127.0.0.1, not localhost: PowerShell 5.1 tries IPv6 first and times out, uvicorn listens on IPv4.
param(
    [int]$Limit = 10,
    [string]$Api = "http://127.0.0.1:8000"
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$backend = Join-Path $root "backend"
$archer = (Resolve-Path (Join-Path $root "..\cuda-archer")).Path
$server = $null
$watch = [Diagnostics.Stopwatch]::StartNew()

function Wait-Job-Result([string]$jobId, [int]$timeoutS = 300) {
    $deadline = (Get-Date).AddSeconds($timeoutS)
    while ((Get-Date) -lt $deadline) {
        $job = Invoke-RestMethod "$Api/api/jobs/$jobId"
        if ($job.status -eq "done") { return $job.result_id }
        if ($job.status -eq "error") { throw "job $jobId failed: $($job.error)" }
        Start-Sleep -Milliseconds 500
    }
    throw "job $jobId timed out after $timeoutS s"
}

function Step([string]$name) {
    Write-Host ("[{0,5:N1}s] {1}" -f $watch.Elapsed.TotalSeconds, $name)
}

try {
    try { $health = Invoke-RestMethod "$Api/api/health" -TimeoutSec 2 }
    catch {
        Step "starting backend"
        $env:VIRTUAL_ENV = $null
        $serverLog = Join-Path $env:TEMP "evalgraph-smoke-api.log"
        $server = Start-Process -FilePath "uv" -ArgumentList "run", "--no-sync", "uvicorn", "evalgraph.api.main:app", "--port", "$(([uri]$Api).Port)" `
            -WorkingDirectory $backend -PassThru -WindowStyle Hidden -RedirectStandardError $serverLog -RedirectStandardOutput "$serverLog.out"
        $deadline = (Get-Date).AddSeconds(60)
        while ($true) {
            try { $health = Invoke-RestMethod "$Api/api/health" -TimeoutSec 2; break }
            catch {
                if ($server.HasExited -or (Get-Date) -gt $deadline) { throw "backend did not start: $((Get-Content $serverLog -Tail 5) -join ' | ')" }
                Start-Sleep -Milliseconds 500
            }
        }
    }
    $mode = if ($health.mock) { "MOCK (pseudo judge scores)" } else { "DIAL $($health.model), judge $($health.judge_model)" }
    Step "API up, mode: $mode"

    Step "analyzing $archer"
    $job = Invoke-RestMethod -Method Post "$Api/api/analyze" -ContentType "application/json" -Body (@{ local_path = $archer } | ConvertTo-Json)
    $analysisId = Wait-Job-Result $job.job_id 300
    $analysis = Invoke-RestMethod "$Api/api/analyses/$analysisId"
    Step ("analysis {0}: {1} components, {2} hotspots, {3} recommendations" -f $analysisId, $analysis.graph.nodes.Count, $analysis.hotspots.Count, $analysis.recommendations.Count)
    if (-not $analysis.summary.runnable) { throw "CUDA Archer analysis is not runnable" }

    $attachments = @($analysis.recommendations | ForEach-Object { @{ node_id = $_.node_id; evaluator_id = $_.evaluator_id } })
    $runs = @{}
    foreach ($variant in @("baseline", "improved")) {
        Step "running $variant (limit $Limit, $($attachments.Count) attachments)"
        $body = @{ analysis_id = $analysisId; attachments = $attachments; variant = $variant; limit = $Limit } | ConvertTo-Json -Depth 5
        $job = Invoke-RestMethod -Method Post "$Api/api/runs" -ContentType "application/json" -Body $body
        $runId = Wait-Job-Result $job.job_id 600
        $run = Invoke-RestMethod "$Api/api/runs/$runId"
        $scored = @($run.results | Where-Object { $_.status -notin @("na", "error") }).Count
        $errors = @($run.results | Where-Object { $_.status -eq "error" }).Count
        Step ("{0}: {1}/{2} scored, {3} errors, {4} bottlenecks, {5}s, mock={6}" -f $runId, $scored, $run.results.Count, $errors, $run.bottlenecks.Count, $run.duration_s, $run.mock)
        if ($scored -eq 0) { throw "run $runId scored nothing" }
        $runs[$variant] = $runId
    }

    $cmp = Invoke-RestMethod "$Api/api/runs/$($runs.baseline)/compare/$($runs.improved)"
    $better = @($cmp.rows | Where-Object { $_.better -eq $true }).Count
    $worse = @($cmp.rows | Where-Object { $_.better -eq $false }).Count
    Step "compare baseline vs improved: $better better, $worse worse, $($cmp.rows.Count - $better - $worse) unchanged/not comparable"
    $cmp.rows | Where-Object { $_.better -ne $null } | Select-Object node_id, evaluator_id, unit, @{n = 'base'; e = { [math]::Round($_.base, 4) } }, @{n = 'after'; e = { [math]::Round($_.other, 4) } }, @{n = 'delta'; e = { [math]::Round($_.delta, 4) } }, better | Format-Table -AutoSize | Out-String | Write-Host
    Step "SMOKE PASSED"
    exit 0
}
catch {
    Write-Host "SMOKE FAILED: $_" -ForegroundColor Red
    exit 1
}
finally {
    if ($server) { Stop-Process -Id $server.Id -Force -ErrorAction SilentlyContinue; Get-CimInstance Win32_Process -Filter "ParentProcessId=$($server.Id)" | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue } }
}
