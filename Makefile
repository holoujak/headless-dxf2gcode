.PHONY: help install install-dev test lint format clean build docs

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install the package in production mode
	pip install -e .

install-dev: ## Install the package with development dependencies
	pip install -r requirements-dev.txt
	pip install -e .
	pre-commit install

lint: ## Run code quality checks
	black --check .
	isort --check-only .
	flake8 .
	@command -v mypy >/dev/null 2>&1 && mypy headless-dxf2gcode.py --ignore-missing-imports || echo "MyPy not installed, skipping type checking"

format: ## Format code automatically
	black .
	isort .

pre-commit: ## Run pre-commit hooks on all files
	pre-commit run --all-files

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -f .coverage
	rm -f test_*.dxf test_*.gcode

build: ## Build the package
	python -m build

docs: ## Generate documentation (placeholder)
	@echo "Documentation generation not implemented yet"

demo: ## Run demo conversion
	@echo "Demo: Converting test_jerky.dxf to G-code"
	python3 headless-dxf2gcode.py test_jerky.dxf demo_output.gcode
	@echo "Demo completed! Check demo_output.gcode"

setup-git: ## Setup git hooks and configuration
	git config core.autocrlf input
	git config pull.rebase false
	pre-commit install --install-hooks

check: ## Run full check suite (lint + test)
	$(MAKE) lint
	$(MAKE) test
	@echo "✅ All checks passed!"

dev-setup: ## Complete development setup
	$(MAKE) install-dev
	$(MAKE) setup-git
	@echo "🎉 Development environment ready!"
