# Makefile for Synthetic Data Generator

.PHONY: help build run status clean test docker-build docker-run docker-status

# Default target
help:
	@echo "Synthetic Data Generator - Available Commands:"
	@echo ""
	@echo "Local Commands:"
	@echo "  make install          - Install dependencies"
	@echo "  make run              - Run processing (set FILE=path/to/file.csv)"
	@echo "  make status           - Check status (set FILE=path/to/file.csv)"
	@echo "  make clean            - Clean temporary files"
	@echo ""
	@echo "Docker Commands:"
	@echo "  make docker-build     - Build Docker image"
	@echo "  make docker-run       - Run with Docker (set FILE=path/to/file.csv)"
	@echo "  make docker-status    - Check status with Docker (set FILE=path/to/file.csv)"
	@echo "  make docker-shell     - Open interactive shell in container"
	@echo "  make docker-clean     - Remove Docker containers and images"

# Local commands
install:
	pip install -r requirements.txt

run:
	@if [ -z "$(FILE)" ]; then \
		echo "Usage: make run FILE=data/input/file.csv"; \
		exit 1; \
	fi
	python -m src.cli.main --config configs/default.yaml $(FILE)

status:
	@if [ -z "$(FILE)" ]; then \
		echo "Usage: make status FILE=data/input/file.csv"; \
		exit 1; \
	fi
	python -m src.cli.status $(FILE)

clean:
	find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.tmp" -delete
	@echo "Cleaned temporary files"

# Docker commands
docker-build:
	docker build -t synthetic-data-generator:latest .

docker-run:
	@if [ -z "$(FILE)" ]; then \
		echo "Usage: make docker-run FILE=data/input/file.csv"; \
		exit 1; \
	fi
	docker-compose run --rm synthetic-data-generator \
		python -m src.cli.main --config configs/default.yaml $(FILE)

docker-status:
	@if [ -z "$(FILE)" ]; then \
		echo "Usage: make docker-status FILE=data/input/file.csv"; \
		exit 1; \
	fi
	docker-compose run --rm synthetic-data-generator \
		python -m src.cli.status $(FILE)

docker-shell:
	docker-compose run --rm synthetic-data-generator /bin/bash

docker-clean:
	docker-compose down
	docker rmi synthetic-data-generator:latest 2>/dev/null || true
	@echo "Cleaned Docker resources"

