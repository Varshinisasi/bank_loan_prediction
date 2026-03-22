import argparse
import json
import ast
from pathlib import Path
from typing import Optional

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


def parse_args():
    parser = argparse.ArgumentParser(description="Bank loan prediction training pipeline")
    parser.add_argument(
        "--data",
        default=r"c:\Users\Jayasrirani S\Downloads\loan_data.csv",
        help="Path to the CSV dataset",
    )
    parser.add_argument("--target", default="Loan_Status", help="Target column name")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set proportion")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", default="outputs", help="Directory for saved results")
    parser.add_argument(
        "--predict-json",
        default=None,
        help="JSON string with applicant details to predict after training",
    )
    parser.add_argument(
        "--predict-file",
        default=None,
        help="Path to a JSON file with applicant details to predict after training",
    )
    parser.add_argument(
        "--model-path",
        default=r"outputs\best_model.joblib",
        help="Path to the saved best model for prediction mode",
    )
    return parser.parse_args()


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def prepare_loan_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Loan_ID" in df.columns:
        df = df.drop(columns=["Loan_ID"])

    if "Dependents" in df.columns:
        dependents = df["Dependents"]
        df["Dependents"] = dependents.where(dependents.isna(), dependents.astype(str).str.strip())
        df["Dependents"] = df["Dependents"].replace("3+", "3")

    categorical_columns = [
        "Gender",
        "Married",
        "Dependents",
        "Education",
        "Self_Employed",
        "Property_Area",
        "Loan_Status",
    ]
    for column in categorical_columns:
        if column in df.columns:
            df[column] = df[column].where(df[column].isna(), df[column].astype(str).str.strip())

    target_mapping = {"Y": 1, "N": 0}
    if "Loan_Status" in df.columns:
        df["Loan_Status"] = df["Loan_Status"].where(
            df["Loan_Status"].isna(), df["Loan_Status"].astype(str).str.strip().str.upper()
        )
        df["Loan_Status"] = df["Loan_Status"].map(target_mapping)

    return df


def create_eda_plots(df: pd.DataFrame, target: str, output_dir: Path) -> None:
    numeric_columns = [
        column
        for column in [
            "ApplicantIncome",
            "CoapplicantIncome",
            "LoanAmount",
            "Loan_Amount_Term",
            "Credit_History",
        ]
        if column in df.columns
    ]

    for column in numeric_columns:
        plt.figure(figsize=(7, 4))
        sns.histplot(df[column].dropna(), kde=True, bins=30, color="#2a9d8f")
        plt.title(f"{column} Distribution")
        plt.xlabel(column)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(output_dir / f"eda_{column.lower()}_distribution.png", dpi=200)
        plt.close()

    if set(["ApplicantIncome", "LoanAmount"]).issubset(df.columns):
        plt.figure(figsize=(7, 5))
        sns.scatterplot(
            data=df,
            x="ApplicantIncome",
            y="LoanAmount",
            hue=target if target in df.columns else None,
            alpha=0.8,
        )
        plt.title("Applicant Income vs Loan Amount")
        plt.tight_layout()
        plt.savefig(output_dir / "eda_income_vs_loan_amount.png", dpi=200)
        plt.close()

    corr_df = df.select_dtypes(include=[np.number]).copy()
    if not corr_df.empty:
        plt.figure(figsize=(9, 7))
        sns.heatmap(corr_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", square=True)
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(output_dir / "eda_correlation_heatmap.png", dpi=200)
        plt.close()


def build_preprocessor(X: pd.DataFrame):
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor


def evaluate_model(name, model, X_test, y_test, output_dir: Path):
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    else:
        y_score = None

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
    }
    if y_score is not None and len(np.unique(y_test)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_test, y_score)
    else:
        metrics["roc_auc"] = np.nan

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_dir / f"confusion_matrix_{name.lower().replace(' ', '_')}.png", dpi=200)
    plt.close()

    print(f"\n{name}")
    print(classification_report(y_test, y_pred, zero_division=0))
    print(json.dumps(metrics, indent=2))
    return metrics, y_score


def tune_model(name, pipeline, param_grid, X_train, y_train):
    search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print(f"\nBest params for {name}: {search.best_params_}")
    print(f"Best CV f1 for {name}: {search.best_score_:.4f}")
    return search.best_estimator_, search.best_params_, search.best_score_


def load_prediction_payload(args) -> Optional[dict]:
    if args.predict_file:
        with open(args.predict_file, "r", encoding="utf-8") as handle:
            return json.load(handle)
    if args.predict_json:
        try:
            return json.loads(args.predict_json)
        except json.JSONDecodeError:
            payload = ast.literal_eval(args.predict_json)
            if not isinstance(payload, dict):
                raise ValueError("Prediction input must be a JSON object or dictionary-like mapping.")
            return payload
    return None


def train_and_save_model(args) -> tuple[Path, list[str]]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(args.data)
    df = prepare_loan_dataset(df)
    if args.target not in df.columns:
        raise ValueError(
            f"Target column '{args.target}' not found. Available columns: {list(df.columns)}"
        )

    df = df.dropna(subset=[args.target]).copy()
    y = df[args.target].astype(int)
    X = df.drop(columns=[args.target])

    if y.nunique() != 2:
        raise ValueError(
            "This script expects a binary classification target. "
            f"Found {y.nunique()} classes after cleaning."
        )

    class_counts = y.value_counts().sort_index()
    plt.figure(figsize=(6, 4))
    sns.barplot(x=class_counts.index.astype(str), y=class_counts.values, palette="viridis")
    plt.title("Target Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_dir / "class_distribution.png", dpi=200)
    plt.close()

    create_eda_plots(df, args.target, output_dir)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    preprocessor = build_preprocessor(X_train)

    models = {
        "Logistic Regression": (
            Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", LogisticRegression(max_iter=1000)),
                ]
            ),
            {
                "model__C": [0.1, 1.0, 10.0],
                "model__solver": ["liblinear"],
            },
        ),
        "Decision Tree": (
            Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", DecisionTreeClassifier(random_state=args.random_state)),
                ]
            ),
            {
                "model__max_depth": [3, 5, 10, None],
                "model__min_samples_split": [2, 5, 10],
                "model__min_samples_leaf": [1, 2, 4],
            },
        ),
        "Random Forest": (
            Pipeline(
                steps=[
                    ("preprocessor", preprocessor),
                    ("model", RandomForestClassifier(random_state=args.random_state)),
                ]
            ),
            {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 5, 10],
                "model__min_samples_split": [2, 5],
            },
        ),
    }

    all_metrics = []
    roc_curves = []
    best_model = None
    best_name = None
    best_f1 = -1

    for name, (pipeline, param_grid) in models.items():
        tuned_model, best_params, best_cv_f1 = tune_model(
            name, pipeline, param_grid, X_train, y_train
        )
        metrics, y_score = evaluate_model(name, tuned_model, X_test, y_test, output_dir)
        metrics["best_params"] = json.dumps(best_params)
        metrics["cv_f1"] = best_cv_f1
        all_metrics.append(metrics)

        if y_score is not None:
            fpr, tpr, _ = roc_curve(y_test, y_score)
            roc_curves.append((name, fpr, tpr, metrics["roc_auc"]))

        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            best_model = tuned_model
            best_name = name

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(output_dir / "metrics.csv", index=False)

    melted = metrics_df.melt(
        id_vars=["model"], value_vars=["accuracy", "precision", "recall", "f1_score"]
    )
    plt.figure(figsize=(10, 6))
    sns.barplot(data=melted, x="model", y="value", hue="variable")
    plt.title("Model Metric Comparison")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.xticks(rotation=15)
    plt.tight_layout()
    plt.savefig(output_dir / "metric_comparison.png", dpi=200)
    plt.close()

    if roc_curves:
        plt.figure(figsize=(7, 6))
        for name, fpr, tpr, roc_auc in roc_curves:
            label = f"{name} (AUC = {roc_auc:.3f})" if not np.isnan(roc_auc) else name
            plt.plot(fpr, tpr, label=label)
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curves")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "roc_curves.png", dpi=200)
        plt.close()

    model_path = output_dir / "best_model.joblib"
    joblib.dump(best_model, model_path)

    print(f"\nBest model saved: {best_name}")
    print(f"Metrics saved to: {output_dir / 'metrics.csv'}")
    print(f"Plots saved to: {output_dir}")
    return model_path, X.columns.tolist()


def predict_new_applicant(model, payload: dict, feature_columns: list[str]) -> None:
    sample = pd.DataFrame([{column: payload.get(column, np.nan) for column in feature_columns}])
    sample = prepare_loan_dataset(sample)
    sample = sample.reindex(columns=feature_columns)
    prediction = int(model.predict(sample)[0])
    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(sample)[0][1])

    approval = "Approved" if prediction == 1 else "Not Approved"
    print("\nPrediction for input applicant:")
    print(json.dumps(payload, indent=2))
    print(f"Predicted result: {approval}")
    if probability is not None:
        print(f"Approval probability: {probability:.4f}")


def main():
    args = parse_args()
    prediction_payload = load_prediction_payload(args)

    if prediction_payload is not None:
        model_path = Path(args.model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Saved model not found at {model_path}. Train the model first, then try prediction."
            )
        model = joblib.load(model_path)
        feature_columns = list(getattr(model, "feature_names_in_", []))
        if not feature_columns:
            df = prepare_loan_dataset(load_data(args.data))
            if args.target in df.columns:
                feature_columns = df.drop(columns=[args.target]).columns.tolist()
            else:
                feature_columns = df.columns.tolist()
        predict_new_applicant(model, prediction_payload, feature_columns)
        return

    train_and_save_model(args)


if __name__ == "__main__":
    main()
