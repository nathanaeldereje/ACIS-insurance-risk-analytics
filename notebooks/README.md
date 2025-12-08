
### notebooks/README.md
```markdown
# Notebooks – Exploratory Analysis & Visualizations

Jupyter notebooks for exploration, prototyping, hypothesis testing, and final stakeholder-ready visualizations.

| Notebook                            | Purpose                                                                                   |
|-------------------------------------|---------------------------------------------------------------------------------------------------|
| 01_data_understanding_eda.ipynb    | Initial data loading, quality checks, descriptive stats, beautiful univariate/bivariate plots    |
| 03_hypothesis_testing.ipynb         | Full A/B testing suite: provinces, postal codes, gender, margins + p-values + business impact   |
| 03_feature_engineering.ipynb        | Creating risk ratios, loss ratios, vehicle age, geographic features, etc.                     |
| 04_modeling_claim_severity.ipynb    | Linear → Random Forest → XGBoost for TotalClaims (where claim > 0) + RMSE & R² comparison       |
| 05_claim_probability_model.ipynb   | Binary classification (claim/no-claim) → probability output                                     |
| 06_risk_based_pricing_model.ipynb  | Combines probability × severity + expense loading → optimal premium prediction                 |
| 07_shap_interpretability.ipynb      | Global & local SHAP explanations → top 10 risk drivers with business translation              |
| 08_final_insights_report.ipynb      | Final plots, cross-segment comparison, executive summary (used for Medium-style final report)  |