# ACIS Insurance Risk Analytics & Predictive Pricing (South Africa)  
**AlphaCare Insurance Solutions – Motor Insurance Analytics Challenge | Dec 2025**

End-to-end risk analytics project to identify low-risk customer segments, statistically validate risk drivers, and build a predictive pricing engine that enables targeted premium reductions for the South African car insurance market.

### Business Goal
Help AlphaCare Insurance Solutions (ACIS) attract profitable new customers by:
- Discovering low-risk provinces, postal codes, driver profiles, and vehicle types
- Proving where statistically significant risk differences exist (and where they don’t)
- Building interpretable ML models for claim prediction and optimal risk-based premiums

### Key Deliverables
- Comprehensive EDA + beautiful, insight-driven visualizations  
- Rigorous A/B hypothesis testing (provinces, postal codes, gender, margins)  
- Reproducible data pipeline with **DVC** (Data Version Control)  
- Claim probability + severity models → risk-based premium framework  
- XGBoost + SHAP explanations for top risk drivers  
- Concrete marketing & pricing recommendations backed by p-values and feature impacts  

### Project Structure
```bash
ACIS-insurance-risk-analytics/
├── data/                  # Raw + versioned datasets (tracked with DVC)
├── notebooks/             # Exploratory analysis & final visualizations
├── scripts/               # Clean, modular production pipeline
├── reports/               # Interim + final report (PDF/Medium-style)
├── sql/                   # Optional queries & aggregations
├── .dvc/                  # DVC configuration & cache
├── .gitignore
├── requirements.txt
└── README.md
```


### Tech Stack
Python • Pandas • Scikit-learn • XGBoost • SHAP • Matplotlib/Seaborn • Plotly • DVC • GitHub Actions • Jupyter

### Quick Start
```bash
git clone https://github.com/nathanaeldereje/ACIS-insurance-risk-analytics.git
cd ACIS-insurance-risk-analytics
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
dvc pull                   # pulls versioned data
jupyter notebook           # explore notebooks/
```
Current Progress (Dec 2025)

Task 1: EDA + Stats → In progress
Task 2: DVC Pipeline → 
Task 3: Hypothesis Testing → 
Task 4: Predictive Modeling + SHAP → 

Challenge completed – Dec _ 2025
Built by Nathanael Dereje