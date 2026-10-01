param(
    [ValidateSet('all', 'english', 'multilingual', 'typed-decisions')][string]$Model = 'all',
    [ValidateSet('cpu', 'cuda')][string]$Device = 'cpu',
    [switch]$Install
)
$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    if (!(Test-Path '.venv\Scripts\python.exe')) {
        & py -3.12 -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Install Python 3.12 (64-bit) with its py launcher, then rerun.' }
    }
    $Python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if ($Install) {
        # Use the user's existing pip configuration, including company Artifactory.
        & $Python -m pip install -r requirements.txt
        if ($LASTEXITCODE -ne 0) { throw 'Runtime dependency installation failed' }
        & $Python -m pip check
        if ($LASTEXITCODE -ne 0) { throw 'Dependency compatibility check failed' }
    }
    & $Python restore.py --model $Model
    if ($LASTEXITCODE -ne 0) { throw 'Model restoration failed' }
    $Checkpoints = if ($Model -eq 'all') { @('english', 'multilingual', 'typed-decisions') } else { @($Model) }
    foreach ($Checkpoint in $Checkpoints) {
        & $Python test_model.py --model $Checkpoint --device $Device --report "reports\$Checkpoint.json"
        if ($LASTEXITCODE -ne 0) { throw "Inference failed: $Checkpoint" }
    }
    & $Python restore.py --model $Model --verify-only
    if ($LASTEXITCODE -ne 0) { throw 'Final snapshot verification failed' }
    Write-Host 'PASS: Restoration and offline inference completed.' -ForegroundColor Green
} finally {
    Pop-Location
}
