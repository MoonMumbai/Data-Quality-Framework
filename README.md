🧠 ML-Guided Data Quality Pattern Selection

An Adaptive Data Quality Framework that automatically analyzes tabular datasets and recommends the most effective data cleaning strategy using Machine Learning.

📌 Overview

This project is a Streamlit-based application that:

  Profiles raw datasets using 9 statistical meta-features
  Uses a trained meta-classifier (Random Forest / Gradient Boosting)
  Recommends one of three data quality remediation patterns
  Provides explainability (SHAP + feature importance)
  Validates improvements using before/after metrics + model performance
  <img width="729" height="619" alt="Screenshot 2026-04-14 at 9 55 51 PM" src="https://github.com/user-attachments/assets/a87ccf24-9c29-4d20-ab2a-d5da4cfd7a4b" />


📄 Based on the research report:
“ML-Guided Data Quality Pattern Selection”

🚀 Features
🔍 1. Automated Data Profiling

  Extracts 9 meta-features:
  
  Missing ratio
  Duplicate ratio
  Outlier ratio
  Entropy
  Skewness
  Numeric ratio
  Unique ratio
  Type consistency
  Value range ratio
🎯 2. ML-Based Pattern Recommendation

  Predicts the best cleaning strategy:
  
  Internal Deficit → Missing values → Imputation
  Interpretation Deficit → Schema issues → Type fixing + scaling
  Operation Deficit → Noise → Deduplication + outlier capping
📊 3. Explainability
  SHAP values for instance-level explanation
  Feature importance (MDI) for global understanding
  Natural language explanation of predictions
📈 4. Validation & Evaluation
  Before vs After quality comparison
  Downstream model evaluation:
  Logistic Regression
  Random Forest
  Decision Tree
  Ablation study:
  No cleaning
  Basic cleaning
ML-recommended cleaning
⚙️ 5. Multi-Model Support
Random Forest (default)
Gradient Boosting (for comparison)
📄 6. Report Generation
Export results as structured JSON report

7)Project Structure

  ├── app_extended.py     # Streamlit frontend (main app) :contentReference[oaicite:1]{index=1}
  ├── core.py             # Backend logic (ML + feature extraction) :contentReference[oaicite:2]{index=2}
  ├── extensions.py       # Advanced features (evaluation, ablation, reports) :contentReference[oaicite:3]{index=3}
  ├── DQ_Framework_MUJ_Report.docx  # Research report  


8) Intallations 

# Clone the repo
`git clone <your-repo-link>`
`cd <repo-folder>`

# Create virtual environment (recommended)
`python -m venv venv`
`source venv/bin/activate`   # Mac/Linux
`venv\Scripts\activate`      # Windows

# Install dependencies
`pip install -r requirements.txt`

9) Usage
`streamlit run app_extended.py`
