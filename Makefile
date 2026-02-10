.PHONY: install dev test lint format backtest paper live data clean

install:
	pip install -r requirements.txt

dev:
	pip install -e ".[dev]"

test:
	python -m pytest tests/ -v --tb=short

lint:
	ruff check src/ tests/

format:
	black src/ tests/

data:
	python -m src.cli.data_cli ingest --input ./data/raw --output ./data/processed --adjust

backtest:
	python -m src.cli.backtest_cli --config ./configs/backtest.yaml

paper:
	python -m src.cli.paper_cli --config ./configs/paper_trade.yaml

live:
	@echo "⚠️  Live trading requires --confirm-live true and .env credentials"
	python -m src.cli.live_cli --config ./configs/live_trade.yaml --confirm-live true

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache dist build *.egg-info
