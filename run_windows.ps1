param(
    [switch]$Smoke,
    [string]$DatasetId = "",
    [string]$RealDataset = "",
    [string]$Config = "configs/default.yaml",
    [switch]$SkipEncode,
    [switch]$SkipFusion
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
Set-Location $PSScriptRoot

function Write-Section([string]$Text) {
    Write-Host ""
    Write-Host "====== $Text"
}

function Invoke-PythonStep([string]$Name, [string[]]$Arguments) {
    $logPath = Join-Path "logs" "${Name}_${script:Stamp}.log"
    Write-Section "[$Name] $($Arguments -join ' ')"
    & $script:Python @Arguments 2>&1 | Tee-Object -FilePath $logPath
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "L'étape '$Name' a échoué (code $exitCode). Log : $logPath"
    }
}

# La modification du power limit NVIDIA demande généralement des droits élevés.
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$principal = [Security.Principal.WindowsPrincipal]::new($identity)
$adminRole = [Security.Principal.WindowsBuiltInRole]::Administrator
if (-not $principal.IsInRole($adminRole)) {
    throw "Ouvre PowerShell avec 'Exécuter en tant qu'administrateur', puis relance cette commande."
}

New-Item -ItemType Directory -Force -Path "logs", "results" | Out-Null
$script:Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$venv = ".venv"
$script:Python = Join-Path $venv "Scripts/python.exe"

if (-not (Test-Path $script:Python)) {
    Write-Section "Création de l'environnement Python"
    $venvCreated = $false
    if (Get-Command "py" -ErrorAction SilentlyContinue) {
        foreach ($version in @("3.12", "3.11", "3.10")) {
            & py "-$version" -c "import sys" 2>$null
            if ($LASTEXITCODE -eq 0) {
                & py "-$version" -m venv $venv
                $venvCreated = ($LASTEXITCODE -eq 0)
                break
            }
        }
    }
    if ((-not $venvCreated) -and (Get-Command "python" -ErrorAction SilentlyContinue)) {
        & python -m venv $venv
        $venvCreated = ($LASTEXITCODE -eq 0)
    }
    if (-not $venvCreated) {
        throw "Python 3.10 à 3.12 est requis. Installe Python 3.12 puis relance."
    }
}

& $script:Python -c "import sys; raise SystemExit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)"
if ($LASTEXITCODE -ne 0) {
    throw "Python 3.10 à 3.12 est requis dans .venv."
}
Write-Host "Python : $(& $script:Python -c 'import sys; print(sys.version.split()[0])')"

$cudaIndex = "https://download.pytorch.org/whl/cu128"
$requirementsHash = (Get-FileHash "requirements_wm.txt" -Algorithm SHA256).Hash
$wantedStamp = "$requirementsHash-$cudaIndex-windows"
$installStamp = Join-Path $venv ".installed"
$alreadyInstalled = (Test-Path $installStamp) -and ((Get-Content $installStamp -Raw).Trim() -eq $wantedStamp)

if (-not $alreadyInstalled) {
    Write-Section "Installation des dépendances CUDA"
    & $script:Python -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "Mise à jour de pip échouée." }
    & $script:Python -m pip install --upgrade --force-reinstall `
        torch==2.10.0 torchvision==0.25.0 --index-url $cudaIndex
    if ($LASTEXITCODE -ne 0) { throw "Installation de PyTorch CUDA échouée." }
    & $script:Python -m pip install -r requirements_wm.txt
    if ($LASTEXITCODE -ne 0) { throw "Installation des dépendances échouée." }
    & $script:Python -m pip install "lerobot>=0.5" -c requirements_wm.txt
    if ($LASTEXITCODE -ne 0) { throw "Installation de LeRobot échouée." }
    Set-Content -Path $installStamp -Value $wantedStamp
}

if (-not (Get-Command "ffmpeg" -ErrorAction SilentlyContinue)) {
    throw "ffmpeg est requis et doit être accessible dans PATH."
}
if (-not (Get-Command "nvidia-smi" -ErrorAction SilentlyContinue)) {
    throw "nvidia-smi est introuvable. Installe ou mets à jour le pilote NVIDIA."
}

if (-not $env:HF_TOKEN -and (Test-Path "configs/hf_token.txt")) {
    $env:HF_TOKEN = (Get-Content "configs/hf_token.txt" -Raw).Trim()
}

if (-not $DatasetId) {
    $DatasetId = (& $script:Python -c `
        "import sys,yaml; print(yaml.safe_load(open(sys.argv[1], encoding='utf-8'))['dataset']['hf_id'])" `
        $Config).Trim()
}

$encoded = "results/encoded/encoded_data.pt"
$realEncoded = "results/encoded/real_encoded_data.pt"
$checkpointSim = "results/checkpoints/predictor_simu.pt"
$checkpointReal = "results/checkpoints/predictor_real.pt"
$encodeExtra = @()
$fusionExtra = @()
$trainExtra = @("--n-epochs", "30", "--batch-size", "64")
$loraExtra = @("--lora-rank", "8", "--n-epochs", "20", "--batch-size", "32")
$demoExtra = @("--n-samples", "1000", "--rollout-chunk", "500")

if ($Smoke) {
    $encoded = "results/encoded/_smoke_encoded_data.pt"
    $realEncoded = "results/encoded/_smoke_real_encoded_data.pt"
    $checkpointSim = "results/checkpoints/_smoke_predictor_simu.pt"
    $checkpointReal = "results/checkpoints/_smoke_predictor_real.pt"
    $encodeExtra = @("--max-pairs", "200")
    $fusionExtra = @("--n-epochs", "2")
    $trainExtra = @("--n-epochs", "2", "--batch-size", "16", "--n-layers", "2")
    $loraExtra = @("--lora-rank", "4", "--n-epochs", "2", "--batch-size", "16")
    $demoExtra = @("--n-samples", "64", "--n-iter", "2", "--horizon", "5")
}

Write-Host ""
Write-Host "LeWM-Robot - pipeline Windows natif"
Write-Host "Dataset : $DatasetId"
Write-Host "GPU     : limite stricte à 80 % du TGP (config $Config)"
Write-Host "Smoke   : $Smoke"

Invoke-PythonStep "check_gpu" @(
    "scripts/00_check_gpu.py", "--config", $Config, "--require-cuda")
Invoke-PythonStep "test_encoder" @(
    "scripts/01_test_dinov3.py", "--config", $Config)

if ((-not $SkipEncode) -or (-not (Test-Path $encoded))) {
    Invoke-PythonStep "encode" (@(
        "scripts/02_encode_dataset.py", "--config", $Config,
        "--dataset-id", $DatasetId, "--output", $encoded) + $encodeExtra)
} else {
    Write-Host "Encodage sauté : $encoded"
}

if (-not $SkipFusion) {
    Invoke-PythonStep "fusion" (@(
        "scripts/03_compare_fusion.py", "--config", $Config,
        "--encoded-data", $encoded) + $fusionExtra)
    $fusionJson = Get-Content "results/fusion_comparison.json" -Raw | ConvertFrom-Json
    $fusion = $fusionJson.PSObject.Properties |
        Sort-Object { [double]$_.Value.best_val_mae } |
        Select-Object -First 1 -ExpandProperty Name
} else {
    $fusion = "concat_view"
}
Write-Host "Fusion retenue : $fusion"

Invoke-PythonStep "train" (@(
    "scripts/04_train_predictor.py", "--config", $Config,
    "--encoded-data", $encoded, "--output", $checkpointSim,
    "--fusion", $fusion) + $trainExtra)

$demoCheckpoint = $checkpointSim
if ($RealDataset) {
    if ((-not $SkipEncode) -or (-not (Test-Path $realEncoded))) {
        Invoke-PythonStep "encode_real" (@(
            "scripts/02_encode_dataset.py", "--config", $Config,
            "--dataset-id", $RealDataset, "--output", $realEncoded) + $encodeExtra)
    }
    Invoke-PythonStep "lora" (@(
        "scripts/05_train_lora.py", "--config", $Config,
        "--real-data", $realEncoded, "--predictor-ckpt", $checkpointSim,
        "--output", $checkpointReal) + $loraExtra)
    $demoCheckpoint = $checkpointReal
}

Invoke-PythonStep "demo" (@(
    "scripts/06_inference_demo.py", "--config", $Config,
    "--dataset-id", $DatasetId, "--predictor-ckpt", $demoCheckpoint) + $demoExtra)

Write-Host ""
Write-Host "Pipeline terminé."
Write-Host "Predictor : $checkpointSim"
Write-Host "Logs      : logs/*_$($script:Stamp).log"
