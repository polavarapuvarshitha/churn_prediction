# Telco Customer Churn ML / MLOps Project

This package contains the complete Lab 3, Lab 4, and Lab 5 project structure.

## Setup

Open PowerShell in this project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Put the dataset at:

```text
data\raw\churn.csv
```

Then run from the project root:

```powershell
python .\pipelines\run_lab3_baseline.py
python .\pipelines\run_lab4_tracking.py
python .\pipelines\run_lab5_pipeline.py
```

## Important

Lab 4 uses `cloudpickle` when logging the Random Forest model to MLflow. This avoids the current MLflow/skops trust error involving `sklearn.tree._tree.Tree` when logging a locally trained Random Forest.

The exact metrics depend on the dataset/version used. Do not expect identical metric values unless the same dataset and environment are used.
