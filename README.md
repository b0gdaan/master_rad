# master_rad

Research pipeline for forecasting time-varying intermarket dependency between
Bitcoin and conventional assets.

The target is the rolling Pearson correlation between a base cryptocurrency and
a conventional asset, Fisher-transformed and forecast one step ahead. Ten model
specifications are evaluated under an expanding-window walk-forward, including a
leakage-safe DCC-GARCH(1,1) benchmark. A second stage turns the dependency
forecast into a binary stress classifier.

Code only. Prices are downloaded on first run and cached; all metrics, figures
and tables are generated into `outputs/`.

## Layout

```
thesis_app/
    pipeline.py           data, features, walk-forward, metrics, DM tests, figures
    dcc.py                DCC-GARCH(1,1) log-likelihood and GARCH recursion
    dcc_walk.py           leakage-safe walk-forward wrapper around dcc.py
    signal_layer.py       stress-day classifier and classification metrics
    data_quality.py       missing-value, outlier and coverage diagnostics
    regime_analysis.py    historical regime catalogue and conditional statistics
    notebook_helpers.py   shared plotting style and helpers
main.py                   single pipeline run
run_all.py                pipeline, tests, and optional notebook/LaTeX steps
run_ablation.py           nested feature-group ablation
train_signal_model.py     fits and persists the signal classifier
replot_figures.py         redraws dataset figures from cached data
setup_dirs.py             creates the output tree
tests/test_pipeline.py    regression tests over the generated outputs
config.yaml               experiment configuration
```

## Running

```
pip install -r requirements.txt
python main.py
```

A full run takes roughly two hours on a desktop CPU with a CUDA-capable GPU for
XGBoost; set `xgb_device: cpu` in `config.yaml` if none is available.

```
python run_all.py --skip-latex --skip-notebooks    # pipeline + tests
python run_ablation.py                             # ablation, needs a prior run
python run_all.py --tests-only                     # tests against existing outputs
```

The `--skip-latex` and `--skip-notebooks` flags are required here: the document
sources and analysis notebooks are not part of this repository.

## Output

```
outputs/results/       metrics, bootstrap intervals, DM tests, sensitivity sweeps (CSV)
outputs/predictions/   per-experiment out-of-sample prediction series (CSV)
outputs/tables/        the same tables exported for LaTeX
outputs/figures/       forecast panels, DM heatmaps, diagnostics (PNG)
data/raw/              cached prices
data/processed/        derived log returns
```

## Configuration

`config.yaml` controls the experiment. The values that change the results:

| Key | Meaning |
|---|---|
| `assets`, `base_asset`, `extra_assets` | tickers, resolved through `yfinance` |
| `start_date`, `end_date` | sample bounds |
| `rolling_windows` | correlation window lengths, default 14/30/60/90 |
| `forecast_horizon` | steps ahead, default 1 |
| `use_fisher_transform` | Fisher-z the target |
| `min_train_size`, `refit_every` | walk-forward warm-up and refit interval |
| `dm_nw_lag` | Newey-West lag for the Diebold-Mariano variance |
| `signal_stress_sigma` | stress-day threshold in trailing sigma |
| `signal_probability_threshold` | classifier decision threshold |
| `random_state` | seed, mirrored for XGBoost |
| `n_parallel_workers` | concurrent pair-window experiments |

## Reproducibility

Runs are deterministic given the cached price file and the seed in
`config.yaml`. No step reads system time or an unpinned remote source. XGBoost
on GPU may differ in the last bits across hardware; set `xgb_device: cpu` for
bit-identical output.

## Requirements

Python 3.11+. Pinned versions in `requirements.txt`; the numerical core is
numpy, pandas, scikit-learn, xgboost, arch and statsmodels.
