# Scripts – Production Pipeline

Clean, modular, well-documented Python scripts designed to be run sequentially or orchestrated.

| Script                         | Description                                                                 |
|--------------------------------|---------------------------------------------------------------------------------|
| 01_load_and_clean.py           | Loads raw data, handles missing values, type corrections, basic cleaning      |
| 02_feature_engineering.py      | Creates all derived features (loss ratio, vehicle age, risk flags, etc.)      |
| 03_hypothesis_tests.py         | Runs all statistical tests + saves results to CSV + prints business summary    |
| 04_train_claim_severity.py     | Trains and saves best severity model (XGBoost)                                |
| 05_train_claim_probability.py | Trains and saves binary classification model                                  |
| 06_risk_based_pricing.py       | Generates predicted optimal premiums using combined models                   |
| 07_shap_analysis.py            | Generates SHAP summary plots and saves top features                           |

**Run order** (after `dvc pull`):
```bash
python scripts/01_load_and_clean.py
python scripts/02_feature_engineering.py
python scripts/03_hypothesis_tests.py

# ... continue the pipeline
```