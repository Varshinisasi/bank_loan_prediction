# Loan Risk Intelligence Project

This project implements a complete loan default / approval risk system with:

- data preprocessing
- model comparison and selection
- SMOTE balancing
- fraud and anomaly detection
- explainable AI output
- a Flask dashboard

It is designed to work with a real loan dataset such as the common Kaggle loan approval dataset, but it also includes a built-in demo dataset so the pipeline can run even if you do not provide one yet.

If you want to use the Kaggle notebook you linked, download the dataset CSV from Kaggle first and place it somewhere local on your machine. The notebook link itself is not the raw CSV file.

## Project Structure

- `loan_prediction.py` - command-line training and prediction entrypoint
- `loan_system.py` - reusable ML pipeline, fraud scoring, and explanations
- `app.py` - Flask dashboard
- `templates/index.html` - dashboard UI
- `static/style.css` - dashboard styling
- `static/app.js` - result chart rendering
- `applicant.json` - sample applicant payload

## Features

- Data cleaning and preprocessing
- Missing value handling
- Numeric normalization
- Categorical encoding
- SMOTE balancing
- Feature engineering:
  - `TotalIncome`
  - `Loan_Income_Ratio`
- Input normalization:
  - `ApplicantIncome` divided by `100`
  - `CoapplicantIncome` divided by `100`
  - `LoanAmount` divided by `1000`
- Rule-based prediction override:
  - approve when `Credit_History = 1` and `Loan_Income_Ratio < 0.05`
  - reject when `Loan_Income_Ratio > 0.1`
  - otherwise use the trained model
- Model comparison:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting or XGBoost if installed
- Metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 score
  - ROC-AUC
- Fraud detection:
  - Isolation Forest
  - Local Outlier Factor
- Risk scoring and anomaly detection
- Explainability with SHAP when available, plus heuristic fallback
- Web dashboard for borrower input and prediction output

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

## Train the Model

Run training on your dataset:

```bash
python loan_prediction.py --data path/to/loan_data.csv --target Loan_Status
```

If you do not provide a dataset, the project will use a built-in demo dataset:

```bash
python loan_prediction.py
```

Outputs are saved in `outputs/`:

- `best_model_bundle.joblib`
- `metrics.csv`
- `evaluation_metrics.csv`
- `evaluation_metrics.json`
- `best_model_name.txt`

The dataset files are saved separately in `data/`:

- `data/raw/loan_dataset_raw.csv`
- `data/processed/loan_dataset_clean.csv`
- `data/processed/train_data.csv`
- `data/processed/test_data.csv`

So you get the raw data, cleaned data, and train/test splits as separate CSV files.

## Predict One Applicant

Use the sample JSON file:

```bash
python loan_prediction.py --predict-file applicant.json
```

Or pass a JSON string:

```bash
python loan_prediction.py --predict-json "{\"Gender\":\"Male\",\"Married\":\"No\",\"Dependents\":\"0\",\"Education\":\"Graduate\",\"Self_Employed\":\"No\",\"ApplicantIncome\":4583,\"CoapplicantIncome\":150,\"LoanAmount\":128,\"Loan_Amount_Term\":360,\"Credit_History\":1,\"Property_Area\":\"Rural\"}"
```

## Run the Dashboard

Start the Flask app:

```bash
python app.py
```

Then open the local server shown in the terminal.

The dashboard lets you:

- enter borrower details
- get the prediction result
- see approval probability
- see a risk score
- see fraud/anomaly flags
- inspect explanation factors

## Notes

- If `xgboost` is installed, the gradient boosting module will use it.
- If `SHAP` is installed, the explanation module will use it.
- If optional libraries are missing, the project falls back to safe built-in behavior.
- If the dataset file path is wrong, the trainer will stop with a clear error instead of silently switching to demo data.
- If you previously trained an older bundle, retrain once so the model can learn the new engineered features.
