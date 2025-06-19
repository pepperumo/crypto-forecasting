.PHONY: help venv install lint test train run-dev build start clean

# Default target
help: ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-12s %s\n", $$1, $$2}'

# Variables
PYTHON := python3
PIP := pip
VENV_DIR := .venv
SYMBOL := btc
HORIZON := 7

# Virtual environment setup
venv: ## Create virtual environment and install backend dependencies
	$(PYTHON) -m venv $(VENV_DIR)
	$(VENV_DIR)/Scripts/activate && $(PIP) install -r backend/requirements.txt
	@echo "Virtual environment created. Activate with: .venv/Scripts/Activate.ps1 (Windows) or source .venv/bin/activate (Unix)"

install: ## Install all dependencies (backend + frontend)
	$(PIP) install -r backend/requirements.txt
	cd frontend && npm install

# Code quality
lint: ## Run linting for backend and frontend
	cd backend && python -m flake8 . --max-line-length=88 --exclude=.venv
	cd frontend && npm run lint

test: ## Run tests
	cd backend && python -m pytest tests/ -v
	cd frontend && npm run test

# Training
train: ## Train model (usage: make train SYMBOL=btc HORIZON=7)
	cd backend && python services/train.py --symbol $(SYMBOL) --horizon $(HORIZON)

# Development
run-dev: ## Run development servers concurrently
	@echo "Starting development servers..."
	@echo "Frontend will be available at: http://localhost:5173"
	@echo "Backend will be available at: http://localhost:8000"
	@echo "API docs will be available at: http://localhost:8000/docs"
	cd backend && uvicorn main:app --reload --port 8000 & \
	cd frontend && npm run dev

# Docker
build: ## Build Docker containers
	docker-compose build

start: ## Start Docker containers
	docker-compose up

stop: ## Stop Docker containers
	docker-compose down

# Cleanup
clean: ## Clean up build artifacts and cache
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	cd frontend && rm -rf node_modules/.cache
	cd frontend && rm -rf dist

# Setup for new users
setup: venv install ## Complete setup for new users
	@echo "Setup complete! Run 'make run-dev' to start development servers."
