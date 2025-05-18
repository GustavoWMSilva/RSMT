# Caminho para o Python no ambiente virtual
# PYTHON=./venv/Scripts/python.exe
PYTHON=python
# Parâmetros do modelo
VERSION=0
EPOCH=100
PHASE_MODEL_PATH=save/phase/version_$(VERSION)/checkpoint_epoch$(EPOCH).pth
MANIFOLD_MODEL=m_save_model_$(EPOCH)

all: train_phase add_phase train_manifold train_sampler validate_sampler
# PIP=./venv/Scripts/pip.exe

# setup:
# 	@echo ">>> Instalando dependências no venv..."
# 	$(PIP) install torch==2.2.0+cu118 torchvision==0.17.0+cu118 torchaudio==2.2.0+cu118 --index-url https://download.pytorch.org/whl/cu118
# 	$(PIP) install -r requirements.txt --no-deps

# preprocess:
# 	@echo ">>> Preprocessando dataset..."
# 	$(PYTHON) process_dataset.py --preprocess

train_phase:
	@echo ">>> Treinando modelo de fase..."
	$(PYTHON) process_dataset.py --train_phase_model
	$(PYTHON) train_deephase.py

add_phase:
	@echo ">>> Adicionando vetores de fase ao dataset..."
	$(PYTHON) process_dataset.py --add_phase_to_dataset --model_path $(PHASE_MODEL_PATH)

train_manifold:
	@echo ">>> Treinando manifold..."
	$(PYTHON) process_dataset.py --train_manifold_model
	$(PYTHON) train_styleVAE.py

train_sampler:
	@echo ">>> Treinando sampler..."
	$(PYTHON) process_dataset.py --train_sampler_model
	$(PYTHON) train_transitionNet.py --moe_model $(MANIFOLD_MODEL)

validate_sampler:
	@echo ">>> Validando modelo final e gerando .bvh..."
	$(PYTHON) train_transitionNet.py --test --moe_model $(MANIFOLD_MODEL) --version $(VERSION) --epoch $(EPOCH)
