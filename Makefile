.PHONY: help install update shell test test-cov lint format typecheck check clean

help:
	@echo "Available commands:"
	@echo "  make install     Install dependencies"
	@echo "  make update      Update dependencies"
	@echo "  make shell       Activate poetry shell"
	@echo "  make test        Run tests"
	@echo "  make test-cov    Run tests with coverage"
	@echo "  make lint        Run Ruff linter"
	@echo "  make format      Format code"
	@echo "  make typecheck   Run mypy"
	@echo "  make check       Run all checks"
	@echo "  make clean       Remove caches/build artifacts"

install:
	poetry install

update:
	poetry update

shell:
	poetry shell

test:
	poetry run pytest

test-cov:
	poetry run pytest --cov=mapping_networks --cov-report=term-missing

lint:
	poetry run ruff check .

format:
	poetry run ruff format .

typecheck:
	poetry run mypy reid scripts 

check: lint typecheck test


clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	rm -rf dist build .coverage htmlcov