.PHONY: setup data panel eda validate-styles test lint clean

# Install dependencies via uv (falls back to documented venv workflow if uv
# is unavailable — see README).
setup:
	uv sync

# Download the H&M dataset from Kaggle. Script does not exist yet
# (Phase 1, later step) — target is wired now so `make data` is the single
# entry point once it lands.
data:
	uv run python scripts/download_data.py

# Build the style_key x ISO-week panel (with support filter) from raw + interim data.
panel:
	uv run python -m nss.features.style_panel

# Run panel EDA: basic stats, top-20 overlap, seasonality plot, stockout-signature detection.
eda:
	uv run python -m nss.viz.panel_eda

# Validate style_key semantic coherence: cluster detail_desc sentence embeddings and compare
# against the categorical style_key partition (ARI/NMI). CPU-only, may take a few minutes.
validate-styles:
	uv run python -m nss.features.style_validation

test:
	uv run pytest --cov=src/nss

lint:
	uv run ruff check .
	uv run ruff format --check .

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} +
