.PHONY: help install init test lint format clean dashboard demo

help:
	@echo "BTC Options + Onchain Engine - Makefile Commands"
	@echo ""
	@echo "  make install    - Install dependencies"
	@echo "  make init       - Initialize database"
	@echo "  make test       - Run tests"
	@echo "  make lint       - Run linters"
	@echo "  make format     - Format code"
	@echo "  make dashboard  - Launch dashboard"
	@echo "  make demo       - Run demo pipeline"
	@echo "  make clean      - Clean generated files"

install:
	uv pip install -e ".[dev]"

init:
	btc-engine init-db

test:
	pytest tests/ -v --cov=btc_engine --cov-report=term-missing

lint:
	ruff check src/ tests/
	mypy src/

format:
	black src/ tests/
	ruff check --fix src/ tests/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .coverage htmlcov/

dashboard:
	btc-engine dashboard

demo:
	@echo "Running demo pipeline..."
	btc-engine init-db
	@echo "Demo mode requires API keys. Please configure .env and run:"
	@echo "  btc-engine ingest-deribit"
	@echo "  btc-engine ingest-glassnode --days 30"
	@echo "  btc-engine build-features"
	@echo "  btc-engine dashboard"
