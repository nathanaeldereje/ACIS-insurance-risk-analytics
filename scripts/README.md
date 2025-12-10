# Scripts – Production Pipeline

Clean, modular, well-documented Python scripts designed to be run sequentially or orchestrated.

| Script                         | Description                                                                 |
|--------------------------------|---------------------------------------------------------------------------------|
| load_and_clean.py           | Loads raw data, handles missing values, type corrections, basic cleaning      |
| eda_metrics.py      | Creates all derived features (loss ratio, vehicle age, risk flags, etc.)      |
| 03_hypothesis_tests.py         | Runs all statistical tests + saves results to CSV + prints business summary    |
| visualizations.py         | Generates visuals and saves them as reports    |
| task3_prepare_analysis_table.py         | validates financial data, generates data/processed/analysis_table.parquet (tracked by DVC), creates summary reports in reports/task3/summary_counts.csv    |
| hypothesis/province.py       | H₁ — Risk differences between provinces(Chi-Square + Cramér’s V (frequency), Kruskal-Wallis / ANOVA (severity).)                   |
| hypothesis/zipcode.py           | H₂ — Risk differences between ZIP codes(Mixed-Effects Logistic Regression (frequency), Kruskal-Wallis / ANOVA (severity).)                           |
| hypothesis/zipcode_margin.py     | H₃ — Margin (profitability) differences by ZIP(OLS with clustered robust SE + Wald F-test.)                                |
| hypothesis/gender.py | H₄ — Risk differences between men and women(Two-proportion z-test (frequency), Welch t-test / Mann-Whitney U (severity).)                                  |
| _04_train_claim_severity.py     | Trains and saves best severity model (XGBoost)                                |
| prepare_severity_data.py | Curates severity modeling dataset + engineered variables + removes leverage outliers.                                  |
| severity_pipeline.py       | Full reproducible severity pipeline (split, train, evaluate, persist models + metrics).                   |
| P07_shap_analysis.py            | Generates SHAP summary plots and saves top features                           |

<!-- **Run order** (after `dvc pull`):
```bash
python scripts/01_load_and_clean.py
python scripts/02_feature_engineering.py
python scripts/03_hypothesis_tests.py

# ... continue the pipeline
``` -->