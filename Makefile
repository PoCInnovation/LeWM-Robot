# ═══════════════════════════════════════════════════════════════════════
# LeWM-Robot — raccourcis pour la machine RTX 4090
#
#   make install        venv + torch CUDA + dépendances + lerobot
#   make bench          ★ TOUT-EN-UN : install si besoin + benchmark → estimation du temps d'un run
#                       (équivalent : bash bench.sh — fonctionne depuis un clone vierge)
#   make check          sanity check GPU/encodeur (01)
#   make smoke          pipeline miniature de bout en bout (~2-5 min)
#   make run            pipeline complet (run_local.sh)
#   make test           pytest
#
# Variables surchargeables :  make bench N_EPOCHS=100 DATASET_ID=user/ds
# ═══════════════════════════════════════════════════════════════════════

SHELL := /bin/bash
.DEFAULT_GOAL := help

# Python : le venv du repo s'il existe, sinon python3
VENV      ?= .venv
PYTHON    := $(if $(wildcard $(VENV)/bin/python),$(VENV)/bin/python,python3)
PIP       := $(PYTHON) -m pip
CUDA_INDEX ?= https://download.pytorch.org/whl/cu128

CONFIG     ?= configs/default.yaml
DATASET_ID ?= $(shell $(PYTHON) -c "import yaml;print(yaml.safe_load(open('$(CONFIG)'))['dataset']['hf_id'])" 2>/dev/null || echo divisio74/duck_dataset_v3)
N_EPOCHS   ?= 30
LORA_EPOCHS ?= 20
BATCH_SIZES ?= 32,64,128
BENCH_ARGS ?=

.PHONY: help install venv check bench bench-quick smoke run encode fusion train lora demo test clean-results

help:
	@grep -E '^#   make' Makefile | sed 's/^#   //'

# ── Installation ────────────────────────────────────────────────────────
venv:
	@test -d $(VENV) || python3 -m venv $(VENV)

install: venv
	$(VENV)/bin/python -m pip install --upgrade pip
	$(VENV)/bin/python -m pip install torch torchvision --index-url $(CUDA_INDEX)
	$(VENV)/bin/python -m pip install -r requirements_wm.txt
	$(VENV)/bin/python -m pip install "lerobot>=0.5"
	@$(VENV)/bin/python -c "import torch;print('CUDA:',torch.cuda.is_available(),'|',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
	@echo "→ huggingface-cli login (DINOv3 gated), puis : make bench"

# ── Benchmark (commande unique) ─────────────────────────────────────────
# Délègue à bench.sh : venv + install si besoin, login HF (HF_TOKEN), GPU,
# puis mini-trains chronométrés → results/benchmark.json + logs/benchmark_*.log.
# Utilise les vrais latents si results/encoded/encoded_data.pt existe, et
# mesure le vrai pipeline d'encodage (décodage vidéo) sur $(DATASET_ID).
bench:
	CONFIG=$(CONFIG) DATASET_ID="$(DATASET_ID)" N_EPOCHS=$(N_EPOCHS) \
	    LORA_EPOCHS=$(LORA_EPOCHS) BATCH_SIZES=$(BATCH_SIZES) \
	    BENCH_ARGS="$(BENCH_ARGS)" CUDA_INDEX=$(CUDA_INDEX) bash bench.sh

# Variante sans dataset ni encodeur (latents synthétiques, ~1 min)
bench-quick:
	$(PYTHON) scripts/07_benchmark.py --config $(CONFIG) --skip-encoder \
	    --n-epochs $(N_EPOCHS) --lora-epochs $(LORA_EPOCHS) \
	    --batch-sizes $(BATCH_SIZES) --output results/benchmark.json $(BENCH_ARGS)

# ── Pipeline ────────────────────────────────────────────────────────────
check:
	$(PYTHON) scripts/01_test_dinov3.py --config $(CONFIG)

smoke:
	SMOKE=1 PY_CONFIG=$(CONFIG) DATASET_ID="$(DATASET_ID)" bash run_local.sh

run:
	PY_CONFIG=$(CONFIG) DATASET_ID="$(DATASET_ID)" bash run_local.sh

encode:
	$(PYTHON) scripts/02_encode_dataset.py --config $(CONFIG) --dataset-id "$(DATASET_ID)"

fusion:
	$(PYTHON) scripts/03_compare_fusion.py --config $(CONFIG)

train:
	$(PYTHON) scripts/04_train_predictor.py --config $(CONFIG) \
	    --encoded-data results/encoded/encoded_data.pt --n-epochs $(N_EPOCHS)

lora:
	$(PYTHON) scripts/05_train_lora.py --config $(CONFIG) --n-epochs $(LORA_EPOCHS)

demo:
	$(PYTHON) scripts/06_inference_demo.py --config $(CONFIG) --dataset-id "$(DATASET_ID)"

# ── Tests / nettoyage ───────────────────────────────────────────────────
test:
	$(PYTHON) -m pytest tests/ -q

clean-results:
	rm -rf results/encoded/_smoke_* results/checkpoints/_smoke_* logs/*.log
