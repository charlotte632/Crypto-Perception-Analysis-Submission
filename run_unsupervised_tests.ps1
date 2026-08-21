# Accept either a quick smoke test or the complete unsupervised analysis.
param(
    [ValidateSet("smoke", "full")]
    [string]$Mode = "smoke"
)

# Stop the script immediately if a PowerShell command reports an error.
$ErrorActionPreference = "Stop"

# Find the project folder and the Python executable in the shared virtual environment.
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Python = Join-Path $ProjectDir "..\.venv\Scripts\python.exe"

# Give a clear error if the expected Python environment is unavailable.
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Project Python was not found at $Python"
}

# Run all later commands from the CryptoEmotionAnalysis project folder.
Set-Location -LiteralPath $ProjectDir

# The smoke test uses fewer tweets and checks that the complete workflow operates correctly.
if ($Mode -eq "smoke") {
    & $Python src\unsupervised_bert_clustering.py `
        --limit-rows 100 `
        --fit-samples-per-dataset 50 `
        --stability-runs 5 `
        --output-dir outputs\paper-replication-unsupervised-smoke
} else {
    # The full test uses the intended sample size and analyses every usable dataset row.
    & $Python src\unsupervised_bert_clustering.py `
        --fit-samples-per-dataset 500 `
        --stability-runs 5 `
        --output-dir outputs\paper-replication-unsupervised
}

# Treat a non-zero Python exit code as a failed analysis rather than continuing silently.
if ($LASTEXITCODE -ne 0) {
    throw "Unsupervised $Mode analysis failed with exit code $LASTEXITCODE"
}

# Confirm which version of the analysis completed successfully.
Write-Host "Completed unsupervised $Mode analysis."
