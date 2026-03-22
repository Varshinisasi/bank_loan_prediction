# Bank Loan Prediction

This project trains and evaluates multiple machine learning models for bank loan prediction.

## What it does

- Loads a CSV dataset from Kaggle or any similar source
- Cleans and preprocesses the data
- Encodes categorical features
- Splits data into train and test sets
- Trains and evaluates:
  - Logistic Regression
  - Decision Tree
  - Random Forest
- Reports:
  - Accuracy
  - Precision
  - Recall
  - F1 score
  - Confusion matrix
- Performs hyperparameter tuning with cross-validation
- Saves charts and plots to an output folder
- Adds EDA charts:
  - income distribution
  - loan amount distribution
  - correlation heatmap
- Lets you pass one applicant record and get a prediction back

## Dataset

The script is designed for your loan approval CSV with these columns:

- `Loan_ID`
- `Gender`
- `Married`
- `Dependents`
- `Education`
- `Self_Employed`
- `ApplicantIncome`
- `CoapplicantIncome`
- `LoanAmount`
- `Loan_Amount_Term`
- `Credit_History`
- `Property_Area`
- `Loan_Status`

It treats `Loan_Status` as the target and converts `Y` to `1`, `N` to `0`.

If your dataset uses a different target column, pass it with `--target`.

## Setup

```bash
pip install -r requirements.txt
```

## Run

Train the model first:

```bash
python loan_prediction.py
```

If your target column is not `Loan_Status`:

```bash
python loan_prediction.py --data path/to/your_dataset.csv --target target_column_name
```

Predict later using the saved best model without retraining:

```bash
python loan_prediction.py --predict-json "{\"Gender\":\"Male\",\"Married\":\"Yes\",\"Dependents\":\"1\",\"Education\":\"Graduate\",\"Self_Employed\":\"No\",\"ApplicantIncome\":4583,\"CoapplicantIncome\":1508,\"LoanAmount\":128,\"Loan_Amount_Term\":360,\"Credit_History\":1,\"Property_Area\":\"Rural\"}"
```

## Output

The script creates an `outputs/` folder containing:

- `metrics.csv`
- `best_model.joblib`
- confusion matrix plots
- metric comparison charts
- class distribution chart
- ROC curve plot
- EDA plots such as:
  - `eda_applicantincome_distribution.png`
  - `eda_coapplicantincome_distribution.png`
  - `eda_loanamount_distribution.png`
  - `eda_loan_amount_term_distribution.png`
  - `eda_credit_history_distribution.png`
  - `eda_income_vs_loan_amount.png`
  - `eda_correlation_heatmap.png`

## Predict With Input

You can also use a JSON file:

```bash
python loan_prediction.py --predict-file applicant.json
```

Prediction mode loads `outputs/best_model.joblib` and does not retrain the model or recompute metrics.

The input fields should match the training features:

- `Gender`
- `Married`
- `Dependents`
- `Education`
- `Self_Employed`
- `ApplicantIncome`
- `CoapplicantIncome`
- `LoanAmount`
- `Loan_Amount_Term`
- `Credit_History`
- `Property_Area`

The script prints:

- predicted result: `Approved` or `Not Approved`
- approval probability, if the model supports it
