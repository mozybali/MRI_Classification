# MRI Classification Web Makefile

VENV = .venv
PYTHON = $(VENV)/bin/python
PIP = $(VENV)/bin/pip

.PHONY: run setup migrate shell clean

run:
	$(PYTHON) manage.py runserver 8000

setup:
	python3 -m venv $(VENV)
	$(PIP) install -r requirements-web.txt
	$(PIP) install numpy pandas scipy torch torchvision torchaudio opencv-python scikit-learn xgboost matplotlib seaborn tqdm optuna SimpleITK scikit-image django-crispy-forms python-dotenv whitenoise gunicorn

migrate:
	$(PYTHON) manage.py makemigrations
	$(PYTHON) manage.py migrate

shell:
	$(PYTHON) manage.py shell

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf staticfiles
	rm -rf media/predictions/*
