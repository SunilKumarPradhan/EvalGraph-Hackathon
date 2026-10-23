# Record the demo for replay mode. Run ON VPN with real DIAL settings in backend\.env (EVALGRAPH_MOCK_LLM=0, ARCHER_MOCK_LLM=0).
# Usage:  .\scripts\record_demo.ps1 [-Source https://github.com/SunilKumarPradhan/cuda-archer] [-Limit 0] [-Out fixtures\demo] [-Append] [-AllowMock]
# Writes manifest.json + analyses\ + runs\ (default fixtures\demo). Then start the app with EVALGRAPH_REPLAY=1 to serve it.
# Refuses to record in mock mode unless -AllowMock (a mock recording is for rehearsals only and shows "REPLAY · MOCK").
param(
    [string]$Source = "https://github.com/SunilKumarPradhan/cuda-archer",
    [int]$Limit = 0,
    [string[]]$Variants = @("baseline", "fix-retrieval", "fix-safety", "improved"),
    [string]$Out = "",
    [switch]$Append,
    [switch]$AllowMock
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
if (-not $Out) { $Out = Join-Path $root "fixtures\demo" }
if (-not [IO.Path]::IsPathRooted($Out)) { $Out = Join-Path (Get-Location) $Out }   # resolve before changing directory
$Out = [IO.Path]::GetFullPath($Out)
if (-not $Source.StartsWith("http")) { $Source = (Resolve-Path $Source).Path }
Set-Location (Join-Path $root "backend")
$env:VIRTUAL_ENV = $null

$mock = uv run --no-sync python -c "from evalgraph.settings import get_settings; print(get_settings().mock_llm)"
if ($mock -eq "True" -and -not $AllowMock) {
    Write-Host "backend\.env has EVALGRAPH_MOCK_LLM=1: a recording now would contain canned answers and pseudo scores." -ForegroundColor Red
    Write-Host "Set EVALGRAPH_MOCK_LLM=0 and ARCHER_MOCK_LLM=0 (on VPN), or pass -AllowMock for a rehearsal-only recording."
    exit 1
}

$cmdArgs = @("run", "--no-sync", "python", "-m", "evalgraph.replay", "record", $Source, "--out", $Out, "--variants") + $Variants
if ($Limit -gt 0) { $cmdArgs += @("--limit", "$Limit") }
if ($Append) { $cmdArgs += @("--append") }   # add variants to the existing recording
$watch = [Diagnostics.Stopwatch]::StartNew()
# PowerShell 5.1 turns any stderr line of a native command (e.g. an app's logged warning) into a terminating error
# under "Stop"; the exit code is what matters here.
$ErrorActionPreference = "Continue"
& uv @cmdArgs
if ($LASTEXITCODE -ne 0) { Write-Host "RECORDING FAILED" -ForegroundColor Red; exit 1 }
Write-Host ("recorded in {0:N0} s to {1}. Copy the real numbers from its manifest.json into docs\DEMO-SCRIPT.md." -f $watch.Elapsed.TotalSeconds, $Out)
