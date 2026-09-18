.PHONY: setup data panel eda validate-styles backtest train forecast forecast-diversity exemplars sweep agent-diagram test lint clean

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

# Run the rolling-origin backtest harness (4 causal baselines) and write
# reports/tables/backtest_per_origin.csv + backtest_summary.csv.
backtest:
	uv run python -m nss.models.backtest

# Train + evaluate the LightGBM model via expanding-window walk-forward through the same
# rolling-origin harness (small bounded hyperparameter search, then one honest walk-forward run),
# plus global SHAP feature importance. Writes reports/tables/backtest_per_origin_lightgbm.csv,
# backtest_summary_lightgbm.csv, shap_global_importance.csv, and reports/figures/shap_global_importance.png.
train:
	uv run python -m nss.models.lightgbm_model

# Train the FINAL production model on all available data (wide weekly origin set), forecast
# AW2020 (origin 2020-09-21), apply the top-3/ranks-4-10 selection rule + local SHAP, and write
# reports/tables/top_styles.csv, top_styles_markdown_excluded.csv, top_styles_by_season.csv.
forecast:
	uv run python -m nss.models.final_forecast

# Diversity-constrained reselection (T1 incumbent / T2 emerging winners, final three, and a
# diversity-constrained seasonal bonus v2), built on top of `forecast`'s trained model + ranking.
# Writes reports/tables/top_styles_t1_incumbent.csv, top_styles_t2_emerging.csv,
# top_styles_final_three.csv, top_styles_by_season_v2.csv.
forecast-diversity:
	uv run python -m nss.models.diversity_forecast

# Select 8 best-selling constituent article images per top-3 winning style + 1 random control
# style (seed=42), fetch them on-demand (no bulk download), and write
# reports/tables/exemplar_images.csv for Phase 3 (novelty scoring).
exemplars:
	uv run python -m nss.data.select_exemplars

# ip_adapter_scale novelty/fidelity sweep (task B4): 8 local_sdxl generations at scales 0.2-0.9
# for one winning style, freed-VRAM CLIP scoring against its real reference images, and
# reports/figures/novelty_fidelity_sweep.png + novelty_fidelity_sweep_images.png. GPU required.
sweep:
	uv run python -m nss.generate.scale_sweep

# Render the Track D agent-delegation architecture diagram (orchestrator + 5 sub-agents, critic
# retry loop) to reports/figures/agent_architecture.png. matplotlib-only, no GPU/data required.
agent-diagram:
	uv run python -m nss.viz.agent_architecture

test:
	uv run pytest --cov=src/nss

lint:
	uv run ruff check .
	uv run ruff format --check .

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name "__pycache__" -exec rm -rf {} +
