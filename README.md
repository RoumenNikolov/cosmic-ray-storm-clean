# Geomagnetic Storm Prediction from Cosmic Ray Measurements

Forecasting the $D_{st}$ geomagnetic disturbance index using ground-based cosmic ray neutron
monitor data combined with near-Earth solar wind parameters (NASA OMNI). See the Abstract and
Introduction in `cosmic_ray_storm_prediction_EDA.ipynb` for the full research question and
domain background.

## Repository structure

The project is split across several notebooks, each depending on artifacts produced by the
ones before it. **Run them in this order:**

| # | Notebook | Produces | Consumes | Execution time |
|---|---|---|---|---|
| 1 | `cosmic_ray_storm_prediction_EDA.ipynb` | — (Abstract, Introduction, Prior Work, IDA, Data Cleaning, EDA) | Raw OMNI/LMKS data | not independently timed |
| 2 | `cosmic_ray_storm_prediction_FE.ipynb` | `data/processed/feat_split.parquet`, `models/context_constants.pkl` | Cleaned data from step 1 | not independently timed |
| 3 | `feature_selection.ipynb` | `models/feature_selection_results.pkl` | Outputs of step 2 | not independently timed |
| 4 | `cosmic_ray_storm_prediction_NFG.ipynb` | Selected model/horizon ($h^*$), MODEL_A–D comparison | Outputs of steps 2–3 | not independently timed |
| 5 | `reference_performance_and_hpo.ipynb` | `models/hp_opt_results.pkl` (tuned XGBoost/LightGBM) | Outputs of steps 2–4 | ~80 min (XGBoost search 45.3 min + LightGBM search 34.1 min; dominates the notebook's total runtime) |
| 6 | `cosmic_ray_storm_prediction_ML.ipynb` | Final model, test set evaluation, Conclusion | Outputs of steps 2–5 | ~1.2 min |

**A note on the times above, in the interest of not overstating what's been measured:** rows 5 and 6 are confirmed — row 5 from the actual search timing printed during hyperparameter optimisation, row 6 from a full, independent end-to-end re-execution used to verify this notebook's results during development. Rows 1–4 have not been independently timed by whoever last edited this table; if you run the full pipeline yourself, consider filling these in from your own `Run All`.

Supporting library code lives in `src/` (feature selectors, custom estimators, evaluation
metrics, AR(2) correction, split definitions) and `src/domain_figures.py` /
`split_visualization.py` (illustrative and diagnostic figures). `imiges_generator.ipynb`
generates the domain-illustration figures used in the EDA notebook's Introduction.

## Requirements

Python 3.11 (`scikit-learn==1.9.0`'s own minimum requirement). Dependencies are listed in
`requirements.txt`, with exact versions confirmed by direct `pip freeze` in the environment
this project was developed and executed in:

```bash
pip install -r requirements.txt
```

### `requirements.txt`

All versions below are confirmed by direct `pip freeze` in the environment this project was
developed and executed in (Python 3.11.9) — not inferred, not a separate test environment.

```
numpy==2.4.6
pandas==2.3.3
scipy==1.17.1
matplotlib==3.11.0
seaborn==0.13.2
scikit-learn==1.9.0
xgboost==3.2.0
lightgbm==4.6.0
joblib==1.5.3
mlflow==3.14.0
plotly==6.8.0
statsmodels==0.14.6
pyarrow==24.0.0
kaleido==1.3.0

# Used by SHAPSelector (src/selectors/shap.py) — retained for future use,
# not called anywhere in the current pipeline; needed only if that selector
# is actually invoked.
shap==0.51.0

# Transitive dependency of mlflow.sklearn.log_model() for securely loading
# custom scikit-learn-compatible estimators (XGBoostDst, LightGBMDst, etc.)
# — not imported directly anywhere in this project's own code.
skops==0.14.0
```

Jupyter (`jupyter`, `jupyterlab`, `ipykernel`) is needed to run the notebooks themselves — any
recent version, not pinned here since it doesn't affect the analysis results.

## Data and model artifacts

Large data and model files (`data/processed/feat_split.parquet`,
`models/feature_selection_results.pkl`, `models/hp_opt_results.pkl`, etc.) are tracked with
[DVC](https://dvc.org) rather than committed directly to this Git repository.

```bash
dvc pull
```

pulls them from the configured remote. All notebooks in this repository are submitted with
their outputs already executed and saved — they can be read directly on GitHub without running
any code or pulling any data. `dvc pull` is only needed to re-execute the pipeline from scratch.

## Reproducibility notes

- MLflow tracking is **off by default** (`ENABLE_MLFLOW = False` in `src/config.py`) — no
  `./mlruns` directory is created unless this is explicitly switched to `True`. Two notebooks
  (`reference_performance_and_hpo.ipynb`, `cosmic_ray_storm_prediction_ML.ipynb`) log real
  experiment-tracking data when enabled; the rest either don't use MLflow at all or only
  initialise it without logging anything.
- `n_jobs=1` is used for model fitting in the Validation stage
  (`cosmic_ray_storm_prediction_NFG.ipynb`) after an earlier run-to-run instability was
  observed with `n_jobs=-1`; the root cause was not fully identified (see that notebook's
  Summary section) but confirmed to be resolved by repeated independent execution.
- `reference_performance_and_hpo.ipynb` and `cosmic_ray_storm_prediction_ML.ipynb` continue to
  use `n_jobs=-1` — a controlled test found no evidence that this notebook's specific
  instability was caused by multi-threaded tree construction, and hyperparameter search
  coverage benefits from full parallelism there.
- `reference_performance_and_hpo.ipynb` includes an automated check for a documented
  degenerate hyperparameter configuration (low `learning_rate` with high `n_estimators` under
  inverse-frequency storm weighting); see that notebook for details.
