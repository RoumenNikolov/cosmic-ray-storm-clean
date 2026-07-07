# Geomagnetic Storm Prediction from Cosmic Ray Measurements

Forecasting the $D_{st}$ geomagnetic disturbance index using ground-based cosmic ray neutron
monitor data combined with near-Earth solar wind parameters (NASA OMNI). See the Abstract and
Introduction in `cosmic_ray_storm_prediction_EDA.ipynb` for the full research question and
domain background.

## Repository structure

The project is split across several notebooks, each depending on artifacts produced by the
ones before it. **Run them in this order:**

| #   | Notebook                                | Folder          | Execution time |
| --- | --------------------------------------- | --------------- | -------------- |
| 1   | `cosmic_ray_storm_prediction_EDA.ipynb` | repository root | ~5 min         |
| 2   | `cosmic_ray_storm_prediction_FE.ipynb`  | repository root | ~5 min         |
| 3   | `feature_selection.ipynb`               | `notebooks/`    | ~20 min        |
| 4   | `cosmic_ray_storm_prediction_NFG.ipynb` | repository root | ~5 min         |
| 5   | `reference_performance_and_hpo.ipynb`   | `notebooks/`    | ~80 min        |
| 6   | `cosmic_ray_storm_prediction_ML.ipynb`  | repository root | ~1.2 min       |

Full description of what each notebook produces, consumes, and is for:

**1. `cosmic_ray_storm_prediction_EDA.ipynb`** — repository root.
Domain background, prior work, and first look at the raw data: initial data analysis, cleaning,
and exploratory analysis of the OMNI solar wind and neutron-monitor records.
_Produces:_ Abstract, Introduction, Prior Work, IDA, Data Cleaning, EDA sections.
_Depends on:_ raw OMNI/LMKS data.

**2. `cosmic_ray_storm_prediction_FE.ipynb`** — repository root.
Builds the engineered feature set (lags, rolling stats, cyclical encodings) and the
chronological train/validation/test split used by every notebook after this one.
_Produces:_ `feat_split.parquet`, `context_constants.pkl`.
_Depends on:_ step 1.

**3. `feature_selection.ipynb`** — `notebooks/` subfolder (uses `../data/...`, `../models/...`
relative paths — run it from inside that folder, not the repository root).
Narrows the engineered features down to a working predictor subset, combining LASSO and
ExtraTrees selection with a majority-vote rule across all five forecasting horizons.
_Produces:_ `feature_selection_results.pkl`.
_Depends on:_ step 2.

**4. `cosmic_ray_storm_prediction_NFG.ipynb`** — repository root.
Tests whether neutron-monitor features improve $D_{st}$ forecasting beyond OMNI alone, under
a controlled protocol, and selects the forecasting horizon $h^*$ on that basis.
_Produces:_ selected horizon $h^*$, MODEL_A–D comparison.
_Depends on:_ steps 2–3.

**5. `reference_performance_and_hpo.ipynb`** — `notebooks/` subfolder (same relative-path
convention as step 3 — run it from inside that folder).
Tunes XGBoost and LightGBM hyperparameters at $h^*$ via `RandomizedSearchCV`.
_Produces:_ `hp_opt_results.pkl` (tuned estimators, best params, CV results).
_Depends on:_ steps 2–4.

**6. `cosmic_ray_storm_prediction_ML.ipynb`** — repository root.
Turns the tuned models into an operational forecaster: validates them, compares
storm-weighting strategies, applies an AR(2) residual correction, selects a final
configuration, and evaluates it on the held-out test set.
_Produces:_ final model, test set evaluation, Conclusion.
_Depends on:_ steps 2–5.

**A note on the folder split, for transparency:** steps 3 and 5 use relative paths one level
up from the other four notebooks (`../data/...` instead of `data/...`) — confirmed directly
from their own code, not assumed. This means they need to be run from a `notebooks/`
subfolder, not the repository root where the other four live. This inconsistency is real, not
a documentation gap, and worth fixing (moving both to the root, or moving all six into
`notebooks/` consistently) rather than just working around it here.

**Execution time.** Only two notebooks have been independently timed: step 5 takes roughly
80 minutes (XGBoost search 45.3 min + LightGBM search 34.1 min — this dominates the notebook's
total runtime), and step 6 takes roughly 1.2 minutes, confirmed via a full end-to-end
re-execution during development. Steps 1–4 haven't been independently timed by whoever last
edited this file — if you run the full pipeline yourself, consider filling these in from your
own `Run All`.

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

Large data and model files aren't committed directly to this Git repository. They're available
as four public archives — no AWS account or credentials needed.

**To run the full pipeline from scratch, only one archive is required:**

| Archive                                                                                                                  | Extract to    | Contents                  |
| ------------------------------------------------------------------------------------------------------------------------ | ------------- | ------------------------- |
| [`data_input.zip`](https://ai-and-ml-2026.s3.eu-north-1.amazonaws.com/cosmic-ray-storm-prediction/backup/data_input.zip) | `data/input/` | Raw OMNI/LMKS input files |

Everything else — `data/processed/feat_split.parquet`, everything in `models/`, every figure in
`images/` — is generated by running the notebooks in the order given in Repository Structure
above, starting from this raw input data.

**If you'd rather skip straight to a later notebook** (e.g. go directly to
`cosmic_ray_storm_prediction_ML.ipynb` without re-running EDA/FE/feature selection/Validation/HPO
first), grab the archives with the intermediate results those earlier stages already produced:

| Archive                                                                                                                          | Extract to        | Contents                                                                             |
| -------------------------------------------------------------------------------------------------------------------------------- | ----------------- | ------------------------------------------------------------------------------------ |
| [`data_processed.zip`](https://ai-and-ml-2026.s3.eu-north-1.amazonaws.com/cosmic-ray-storm-prediction/backup/data_processed.zip) | `data/processed/` | `feat_split.parquet` and related processed artifacts                                 |
| [`models.zip`](https://ai-and-ml-2026.s3.eu-north-1.amazonaws.com/cosmic-ray-storm-prediction/backup/models.zip)                 | `models/`         | `context_constants.pkl`, `feature_selection_results.pkl`, `hp_opt_results.pkl`, etc. |
| [`images.zip`](https://ai-and-ml-2026.s3.eu-north-1.amazonaws.com/cosmic-ray-storm-prediction/backup/images.zip)                 | `images/`         | Already-rendered figures, if you just want to view them without regenerating         |

All notebooks are submitted with their outputs already executed and saved — they can be read
directly on GitHub without downloading anything or running any code at all. These archives are
only needed if you want to actually re-execute part or all of the pipeline yourself.

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
  `cosmic_ray_storm_prediction_ML.ipynb` applies the same fix specifically to its two
  from-scratch XGBoost retrains (`XGBoostDstV2` in Storm Weighting Strategy, `XGBoostDst` for
  the MODEL_A baseline in Final Model Selection) — confirmed by direct testing to produce
  bit-identical predictions across separate fits, at negligible cost since these are one-off
  retrains, not repeated hyperparameter searches.
- `reference_performance_and_hpo.ipynb` continues to use `n_jobs=-1` for its `XGBoostDst`
  instances — a controlled test found no evidence that instability was caused by
  multi-threaded tree construction, and the 250-fit hyperparameter search benefits
  substantially from full parallelism there; changing the default would only cost time with
  no established reproducibility benefit for this specific search.
- `reference_performance_and_hpo.ipynb` includes an automated check for a documented
  degenerate hyperparameter configuration (low `learning_rate` with high `n_estimators` under
  inverse-frequency storm weighting); see that notebook for details.
