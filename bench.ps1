# =====================================================================
# Benchmark RTX 5090 sous Windows PowerShell :
#     .\bench.ps1
#
# Fait tout automatiquement :
#   1. Vérifie Python 3.10-3.12
#   2. Crée .venv et installe PyTorch CUDA 12.8 + dépendances
#   3. Charge le token HF depuis configs\hf_token.txt
#   4. Lance l'estimation de temps (scripts\07_benchmark.py)
#   5. Affiche le rapport de synthèse
# =====================================================================

param(
    [string]$Config = "configs/default.yaml",
    [string]$DatasetId = "",
    [int]$NEpochs = 30,
    [int]$LoraEpochs = 20,
    [string]$BatchSizes = "32,64,128,256",
    [switch]$SkipEncoder
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
Set-Location $PSScriptRoot

# 1. Vérification des droits administrateur (recommandé pour power limit NVIDIA)
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = [Security.Principal.WindowsPrincipal]::new($identity)
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    Write-Warning "PowerShell n'est pas en mode administrateur. Le réglage du power limit GPU peut échouer."
}

# 2. Détection / création du venv
$venv = ".venv"
$python = Join-Path $venv "Scripts\python.exe"

if (-not (Test-Path $python)) {
    Write-Host "Création du venv..." -ForegroundColor Cyan
    $created = $false
    if (Get-Command "py" -ErrorAction SilentlyContinue) {
        foreach ($v in @("3.12", "3.11", "3.10")) {
            & py "-$v" -c "import sys" 2>$null
            if ($LASTEXITCODE -eq 0) {
                & py "-$v" -m venv $venv
                $created = ($LASTEXITCODE -eq 0)
                break
            }
        }
    }
    if ((-not $created) -and (Get-Command "python" -ErrorAction SilentlyContinue)) {
        & python -m venv $venv
        $created = ($LASTEXITCODE -eq 0)
    }
    if (-not $created) {
        throw "Python 3.10 à 3.12 est requis. Installez Python 3.12 puis relancez."
    }
}

# 3. Installation des dépendances si nécessaire
$cudaIndex = "https://download.pytorch.org/whl/cu128"
$requirementsHash = (Get-FileHash "requirements_wm.txt" -Algorithm SHA256).Hash
$wantedStamp = "$requirementsHash-$cudaIndex-windows"
$installStamp = Join-Path $venv ".installed"
$alreadyInstalled = (Test-Path $installStamp) -and ((Get-Content $installStamp -Raw).Trim() -eq $wantedStamp)

if (-not $alreadyInstalled) {
    Write-Host "Installation des dépendances (PyTorch CUDA 12.8, LeRobot...)" -ForegroundColor Cyan
    & $python -m pip install --upgrade pip
    & $python -m pip install --upgrade --force-reinstall torch==2.10.0 torchvision==0.25.0 --index-url $cudaIndex
    & $python -m pip install -r requirements_wm.txt
    & $python -m pip install "lerobot==0.5.1" -c requirements_wm.txt
    Set-Content -Path $installStamp -Value $wantedStamp
}

# 4. Token Hugging Face
if (-not $env:HF_TOKEN -and (Test-Path "configs\hf_token.txt")) {
    $env:HF_TOKEN = (Get-Content "configs\hf_token.txt" -Raw).Trim()
}

# 5. Lancement de l'estimation (benchmark)
New-Item -ItemType Directory -Force -Path "logs", "results" | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = Join-Path "logs" "benchmark_$stamp.log"

$benchArgs = @(
    "scripts\07_benchmark.py",
    "--config", $Config,
    "--n-epochs", "$NEpochs",
    "--lora-epochs", "$LoraEpochs",
    "--batch-sizes", $BatchSizes,
    "--output", "results\benchmark.json"
)

if ($SkipEncoder) {
    $benchArgs += "--skip-encoder"
} elseif ($DatasetId) {
    $benchArgs += @("--dataset-id", $DatasetId)
}

Write-Host "`nLancement du benchmark chronométré..." -ForegroundColor Green
& $python @benchArgs 2>&1 | Tee-Object -FilePath $logPath

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nGénération du rapport de synthèse..." -ForegroundColor Green
    & $python "scripts\bench_report.py" "results"
    Write-Host "`n[SUCCÈS] Rapport disponible dans results\benchmark.json et results\summary.md" -ForegroundColor Green
} else {
    Write-Error "Le benchmark s'est arrêté avec le code d'erreur $LASTEXITCODE. Détails dans $logPath"
}
