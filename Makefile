.PHONY: setup data panel eda test lint clean

# Install dependencies via uv (falls back to documented venv workflow if uv
# is unavailable — see README).
setup:
	uv sync

# Download the H&M dataset from Kaggle. Script does not exist yet
# (Phase 1, later step) — target is wired now so `make data` is the single
# entry point once it lands.
data:
	uv run python scripts/download_data.py

# Build the style-week panel from raw data. Script does not exist yet
# (later phase) — wired now for reproducibility.
panel:
	uv run python -m nss.data.build_panel

# Run the EDA pipeline/notebook export. Script does not exist yet
# (later phase) — wired now for reproducibility.
eda:
	uv run python -m nss.viz.eda

test:
	uv run pytest --cov=src/nss

lint:
	uv run ruff check .
	uv run ruff format --check .

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} +
