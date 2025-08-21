# Data Cleaning & Transformation Toolkit (cleanit)

A reusable **Python package** to clean messy datasets: remove duplicates, fill missing values, scale/normalize features, and encode categorical columns. Includes **Grafana dashboard** JSON + **sample screenshots** for monitoring dataset health metrics (missing values, duplicates, feature ranges).

## Features
- `remove_duplicates()` — drop exact/partial duplicates with optional subset keys
- `fill_missing()` — numeric strategies (mean/median/constant), categorical (mode/constant)
- `encode_categoricals()` — one-hot or ordinal with stored mapping for reuse
- `scale_features()` — standard/min-max/robust scaling with fitted scalers
- `report_profile()` — quick table of missing %, distinct counts, numeric stats
- **Pipelines** — `CleanPipeline` to chain steps reproducibly with `.fit()/.transform()`
- **Artifacts** — saves fitted encoders/scalers/mappings for inference
- **Examples** — scripts applying toolkit on **Iris** and **Titanic-like** datasets
- **CI** — GitHub Actions for tests & lint
- **Packaging** — `pyproject.toml` for `pip install -e .`

## Install
```bash
pip install -e .
```

## Usage (Quick Start)
```python
from cleanit.pipeline import CleanPipeline
from cleanit.transforms import remove_duplicates, fill_missing, encode_categoricals, scale_features

pipe = CleanPipeline(steps=[
    remove_duplicates(subset=["id"]),
    fill_missing(num_strategy="median", cat_strategy="mode"),
    encode_categoricals(method="onehot"),
    scale_features(method="standard")
])

df_clean = pipe.fit_transform(df)
pipe.save("artifacts/clean_pipeline.pkl")
```

## Grafana
- Import `grafana/dashboard.json` to visualise data health KPIs.
- Sample screenshots live in `grafana/screenshots/` for your README.

## Repo Layout
```
data-cleaning-toolkit/
├── cleanit/ (package)
├── data/ (sample datasets)
├── examples/ (how to apply on datasets)
├── grafana/ (dashboard + screenshots)
├── tests/
├── .github/workflows/ci.yml
├── pyproject.toml
└── README.md
```
