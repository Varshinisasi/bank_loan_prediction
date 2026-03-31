from __future__ import annotations

import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    IsolationForest,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
except Exception:  # pragma: no cover - optional dependency
    ImbPipeline = None
    SMOTE = None

try:  # pragma: no cover - optional dependency
    import shap  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    shap = None

try:  # pragma: no cover - optional dependency
    from xgboost import XGBClassifier  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    XGBClassifier = None

try:  # pragma: no cover - optional dependency
    import joblib  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    joblib = None


APPROVED_LABELS = {"y", "yes", "approved", "approve", "1", "true", "paid", "repaid"}
REJECTED_LABELS = {"n", "no", "rejected", "reject", "0", "false", "default", "charged_off"}
DEFAULT_TARGET_CANDIDATES = ["Loan_Status", "loan_status", "target", "Default", "default"]


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # pragma: no cover - older scikit-learn
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_target_mapping(values: pd.Series) -> Optional[dict[Any, int]]:
    clean_values = [str(v).strip().lower() for v in values.dropna().unique()]
    if not clean_values:
        return None

    approved_hits = sum(value in APPROVED_LABELS for value in clean_values)
    rejected_hits = sum(value in REJECTED_LABELS for value in clean_values)
    if approved_hits or rejected_hits:
        mapping: dict[Any, int] = {}
        for original in values.dropna().unique():
            label = str(original).strip().lower()
            if label in APPROVED_LABELS:
                mapping[original] = 1
            elif label in REJECTED_LABELS:
                mapping[original] = 0
        return mapping

    numeric_values = sorted({int(float(v)) for v in values.dropna().unique() if _is_number(v)})
    if set(numeric_values).issubset({0, 1}):
        return None
    return None


def _is_number(value: Any) -> bool:
    try:
        float(value)
        return True
    except Exception:
        return False


def normalize_binary_target(series: pd.Series) -> pd.Series:
    mapping = build_target_mapping(series)
    if mapping:
        normalized = series.map(mapping)
    else:
        normalized = series.copy()

    if normalized.dtype == object:
        normalized = normalized.astype(str).str.strip().str.lower()
        normalized = normalized.map(
            lambda value: 1
            if value in APPROVED_LABELS
            else 0
            if value in REJECTED_LABELS
            else value
        )

    if normalized.dropna().isin([0, 1]).all():
        return normalized.astype(int)

    numeric = pd.to_numeric(normalized, errors="coerce")
    if numeric.dropna().isin([0, 1]).all():
        return numeric.astype(int)

    raise ValueError(
        "The target column must be binary and should map cleanly to 0/1 or approved/rejected labels."
    )


def prepare_dataframe(df: pd.DataFrame, target_column: Optional[str] = None) -> pd.DataFrame:
    frame = df.copy()
    frame.columns = frame.columns.str.strip()

    if "Loan_ID" in frame.columns:
        frame = frame.drop(columns=["Loan_ID"])

    if "Dependents" in frame.columns:
        frame["Dependents"] = frame["Dependents"].replace("3+", "3")

    for column in frame.columns:
        if frame[column].dtype == object:
            frame[column] = frame[column].astype(str).str.strip()
            frame[column] = frame[column].replace({"": np.nan, "nan": np.nan, "None": np.nan})

    if target_column and target_column in frame.columns:
        frame[target_column] = normalize_binary_target(frame[target_column])

    numeric_candidates = [
        "ApplicantIncome",
        "CoapplicantIncome",
        "LoanAmount",
        "Loan_Amount_Term",
        "Credit_History",
        "CreditScore",
        "Age",
        "Income",
        "DebtToIncome",
        "LoanToIncome",
    ]
    for column in numeric_candidates:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame


def infer_target_column(df: pd.DataFrame, preferred: Optional[str] = None) -> str:
    if preferred and preferred in df.columns:
        return preferred
    for candidate in DEFAULT_TARGET_CANDIDATES:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        "No target column found. Provide --target explicitly or include one of "
        f"{DEFAULT_TARGET_CANDIDATES}."
    )


def load_dataset(path: Optional[str]) -> pd.DataFrame:
    if path is None:
        return generate_demo_dataset()

    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    if data_path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(data_path)
    if data_path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(data_path)
    raise ValueError(f"Unsupported dataset format: {data_path.suffix}")


def generate_demo_dataset(rows: int = 320, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    genders = np.array(["Male", "Female"])
    married = np.array(["Yes", "No"])
    dependents = np.array(["0", "1", "2", "3+"])
    education = np.array(["Graduate", "Not Graduate"])
    self_employed = np.array(["Yes", "No"])
    property_area = np.array(["Urban", "Semiurban", "Rural"])

    applicant_income = rng.integers(1800, 16000, size=rows)
    coapplicant_income = rng.integers(0, 8000, size=rows)
    loan_amount = rng.integers(50, 700, size=rows)
    loan_term = rng.choice([120, 180, 240, 360, 480], size=rows, p=[0.1, 0.1, 0.2, 0.5, 0.1])
    credit_history = rng.choice([0, 1], size=rows, p=[0.35, 0.65])

    score = (
        0.34 * (applicant_income / applicant_income.max())
        + 0.16 * (coapplicant_income / max(coapplicant_income.max(), 1))
        - 0.30 * (loan_amount / loan_amount.max())
        + 0.24 * credit_history
        + rng.normal(0, 0.08, size=rows)
    )
    approval_probability = 1 / (1 + np.exp(-4 * (score - score.mean())))
    loan_status = np.where(approval_probability > 0.5, "Y", "N")

    frame = pd.DataFrame(
        {
            "Loan_ID": [f"DEMO{i:04d}" for i in range(rows)],
            "Gender": rng.choice(genders, size=rows),
            "Married": rng.choice(married, size=rows),
            "Dependents": rng.choice(dependents, size=rows),
            "Education": rng.choice(education, size=rows),
            "Self_Employed": rng.choice(self_employed, size=rows),
            "ApplicantIncome": applicant_income,
            "CoapplicantIncome": coapplicant_income,
            "LoanAmount": loan_amount,
            "Loan_Amount_Term": loan_term,
            "Credit_History": credit_history,
            "Property_Area": rng.choice(property_area, size=rows),
            "Loan_Status": loan_status,
        }
    )

    mask = rng.random(rows) < 0.04
    frame.loc[mask, "LoanAmount"] = np.nan
    mask = rng.random(rows) < 0.03
    frame.loc[mask, "Self_Employed"] = np.nan
    return frame


def build_preprocessor(X: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_features = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_features = [column for column in X.columns if column not in numeric_features]

    numeric_transformer = SkPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = SkPipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", make_one_hot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor, numeric_features, categorical_features


def make_sampler(y: pd.Series, random_state: int):
    if SMOTE is None:
        return None
    counts = y.value_counts()
    if counts.empty or counts.min() < 2:
        return None
    k_neighbors = min(5, counts.min() - 1)
    if k_neighbors < 1:
        return None
    return SMOTE(random_state=random_state, k_neighbors=k_neighbors)


def build_pipeline(
    preprocessor: ColumnTransformer,
    sampler: Any,
    classifier: Any,
):
    if sampler is not None and ImbPipeline is not None:
        return ImbPipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("sampler", sampler),
                ("model", classifier),
            ]
        )
    return SkPipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", classifier),
        ]
    )


def build_models(random_state: int, preprocessor, sampler) -> dict[str, tuple[Any, dict[str, list[Any]]]]:
    models: dict[str, tuple[Any, dict[str, list[Any]]]] = {
        "Logistic Regression": (
            build_pipeline(
                preprocessor,
                sampler,
                LogisticRegression(max_iter=1500, class_weight="balanced"),
            ),
            {
                "model__C": [0.1, 1.0, 5.0],
            },
        ),
        "Decision Tree": (
            build_pipeline(
                preprocessor,
                sampler,
                DecisionTreeClassifier(random_state=random_state, class_weight="balanced"),
            ),
            {
                "model__max_depth": [3, 5, 10, None],
                "model__min_samples_split": [2, 5, 10],
            },
        ),
        "Random Forest": (
            build_pipeline(
                preprocessor,
                sampler,
                RandomForestClassifier(
                    random_state=random_state,
                    class_weight="balanced",
                    n_estimators=200,
                ),
            ),
            {
                "model__max_depth": [None, 6, 10],
                "model__min_samples_split": [2, 5],
            },
        ),
    }

    if XGBClassifier is not None:
        models["Gradient Boosting"] = (
            build_pipeline(
                preprocessor,
                sampler,
                XGBClassifier(
                    random_state=random_state,
                    n_estimators=150,
                    max_depth=4,
                    learning_rate=0.08,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    eval_metric="logloss",
                    tree_method="hist",
                ),
            ),
            {
                "model__n_estimators": [100, 150],
                "model__max_depth": [3, 4, 5],
                "model__learning_rate": [0.05, 0.08, 0.1],
            },
        )
    else:
        models["Gradient Boosting"] = (
            build_pipeline(
                preprocessor,
                sampler,
                GradientBoostingClassifier(random_state=random_state),
            ),
            {
                "model__n_estimators": [100, 150],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [2, 3],
            },
        )

    return models


def get_model_estimator(bundle: "ModelBundle") -> Any:
    model = bundle.model
    if hasattr(model, "named_steps"):
        return model.named_steps["model"]
    return model


def get_preprocessor(bundle: "ModelBundle") -> ColumnTransformer:
    return bundle.preprocessor


def get_feature_names(bundle: "ModelBundle") -> list[str]:
    names: list[str] = []
    names.extend(bundle.numeric_features)
    encoder = bundle.preprocessor.named_transformers_["cat"].named_steps["onehot"]
    for feature_name, categories in zip(bundle.categorical_features, encoder.categories_):
        names.extend([f"{feature_name}={category}" for category in categories])
    return names


def _format_explanation_items(items: list[tuple[str, float]]) -> list[dict[str, Any]]:
    formatted: list[dict[str, Any]] = []
    for feature, impact in items:
        impact_value = float(impact)
        direction = "pushes toward approval" if impact_value > 0 else "pushes toward rejection"
        formatted.append(
            {
                "feature": feature,
                "impact": round(impact_value, 4),
                "direction": direction,
                "display": f"{feature}: {abs(impact_value):.4f} ({direction})",
            }
        )
    return formatted


def fit_fraud_models(
    bundle: "ModelBundle",
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
) -> dict[str, Any]:
    transformed = bundle.preprocessor.transform(X_train)

    isoforest = IsolationForest(contamination=0.1, random_state=random_state)
    isoforest.fit(transformed)

    lof_neighbors = min(20, max(2, len(X_train) - 1))
    lof = LocalOutlierFactor(n_neighbors=lof_neighbors, novelty=True)
    lof.fit(transformed)

    iso_scores = -isoforest.decision_function(transformed)
    lof_scores = -lof.decision_function(transformed)

    stats = {
        "iso_mean": float(np.mean(iso_scores)),
        "iso_std": float(np.std(iso_scores) + 1e-9),
        "lof_mean": float(np.mean(lof_scores)),
        "lof_std": float(np.std(lof_scores) + 1e-9),
    }

    return {
        "isolation_forest": isoforest,
        "local_outlier_factor": lof,
        "stats": stats,
    }


@dataclass
class ModelBundle:
    model_name: str
    model: Any
    preprocessor: ColumnTransformer
    feature_columns: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    target_column: str
    target_mapping: dict[str, int]
    metrics: list[dict[str, Any]]
    best_threshold: float
    fraud_models: dict[str, Any]
    training_rows: int
    reference_data: Optional[pd.DataFrame] = None
    positive_label: str = "Approved"
    negative_label: str = "Rejected"


def evaluate_pipeline(
    name: str,
    pipeline: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, Any]:
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None
    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": np.nan,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    if y_prob is not None and len(np.unique(y_test)) > 1:
        metrics["roc_auc"] = roc_auc_score(y_test, y_prob)
    return metrics


def tune_model(
    name: str,
    pipeline: Any,
    param_grid: dict[str, list[Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int,
) -> Any:
    search = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=cv_folds,
        scoring="f1",
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_


def train_project(
    data_path: Optional[str],
    target_column: Optional[str],
    output_dir: str = "outputs",
    data_dir: str = "data",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[ModelBundle, pd.DataFrame]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    data_root = Path(data_dir)
    raw_path = data_root / "raw"
    processed_path = data_root / "processed"
    raw_path.mkdir(parents=True, exist_ok=True)
    processed_path.mkdir(parents=True, exist_ok=True)

    raw_df = load_dataset(data_path)
    raw_df.to_csv(raw_path / "loan_dataset_raw.csv", index=False)
    target = infer_target_column(raw_df, preferred=target_column)
    df = prepare_dataframe(raw_df, target)
    df.to_csv(processed_path / "loan_dataset_clean.csv", index=False)
    df = df.dropna(subset=[target]).copy()

    y = normalize_binary_target(df[target])
    X = df.drop(columns=[target])

    if y.nunique() != 2:
        raise ValueError("The selected target column must contain exactly two classes.")

    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError(
            "The train/test split does not contain both classes. "
            "Please provide a larger dataset or adjust the split."
        )

    train_df = X_train.copy()
    train_df[target] = y_train.values
    test_df = X_test.copy()
    test_df[target] = y_test.values
    train_df.to_csv(processed_path / "train_data.csv", index=False)
    test_df.to_csv(processed_path / "test_data.csv", index=False)

    preprocessor, numeric_features, categorical_features = build_preprocessor(X_train)
    preprocessor.fit(X_train)
    sampler = make_sampler(y_train, random_state=random_state)
    cv_folds = max(2, min(5, int(y_train.value_counts().min())))

    models = build_models(random_state, preprocessor, sampler)
    results: list[dict[str, Any]] = []
    best_name: Optional[str] = None
    best_model: Any = None
    best_score = -1.0

    for name, (pipeline, param_grid) in models.items():
        tuned = tune_model(name, pipeline, param_grid, X_train, y_train, cv_folds=cv_folds)
        metrics = evaluate_pipeline(name, tuned, X_test, y_test)
        results.append(metrics)
        if metrics["f1_score"] > best_score:
            best_score = metrics["f1_score"]
            best_name = name
            best_model = tuned

    metrics_df = pd.DataFrame(results).sort_values(by="f1_score", ascending=False)
    metrics_csv = metrics_df.copy()
    if "confusion_matrix" in metrics_csv.columns:
        metrics_csv["confusion_matrix"] = metrics_csv["confusion_matrix"].apply(json.dumps)
    metrics_csv.to_csv(output_path / "metrics.csv", index=False)
    metrics_csv.to_csv(output_path / "evaluation_metrics.csv", index=False)
    (output_path / "evaluation_metrics.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    (output_path / "best_model_name.txt").write_text(best_name or "Unknown", encoding="utf-8")

    bundle = ModelBundle(
        model_name=best_name or "Unknown",
        model=best_model,
        preprocessor=best_model.named_steps["preprocessor"]
        if hasattr(best_model, "named_steps")
        else preprocessor,
        feature_columns=X.columns.tolist(),
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        target_column=target,
        target_mapping={"approved": 1, "rejected": 0},
        metrics=results,
        best_threshold=0.5,
        fraud_models={},
        training_rows=len(X_train),
        reference_data=X_train.sample(n=min(120, len(X_train)), random_state=random_state).copy(),
    )

    bundle.fraud_models = fit_fraud_models(bundle, X_train, y_train, random_state)
    bundle_path = output_path / "best_model_bundle.joblib"
    if joblib is not None:
        joblib.dump(bundle, bundle_path)
    else:
        with bundle_path.open("wb") as handle:
            pickle.dump(bundle, handle)
    return bundle, metrics_df


def load_bundle(model_path: str = "outputs/best_model_bundle.joblib") -> ModelBundle:
    bundle_path = Path(model_path)
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"Model bundle not found at {bundle_path}. Train the project first."
        )
    if joblib is not None:
        return joblib.load(bundle_path)
    with bundle_path.open("rb") as handle:
        return pickle.load(handle)


def _coerce_sample(bundle: ModelBundle, sample: dict[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame([{column: sample.get(column, np.nan) for column in bundle.feature_columns}])
    for column in bundle.numeric_features:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in bundle.categorical_features:
        if column in frame.columns:
            frame[column] = frame[column].astype(str).replace({"nan": np.nan, "None": np.nan})
    if "Dependents" in frame.columns:
        frame["Dependents"] = frame["Dependents"].replace("3+", "3")
    return frame


def _normalize_risk(value: float, mean: float, std: float) -> float:
    if std <= 1e-9:
        return float(np.clip(value, 0.0, 1.0))
    z_score = (value - mean) / std
    return float(1.0 / (1.0 + math.exp(-z_score)))


def _safe_float(value: Any, default: float = 0.0) -> float:
    if pd.isna(value):
        return default
    try:
        return float(value)
    except Exception:
        return default


def _heuristic_risk(bundle: ModelBundle, sample: pd.DataFrame) -> tuple[float, list[dict[str, Any]]]:
    risk_notes: list[dict[str, Any]] = []
    risk = 0.0

    applicant_income = _safe_float(sample.get("ApplicantIncome", pd.Series([np.nan])).iloc[0], 0.0)
    coapplicant_income = _safe_float(sample.get("CoapplicantIncome", pd.Series([np.nan])).iloc[0], 0.0)
    loan_amount = _safe_float(sample.get("LoanAmount", pd.Series([np.nan])).iloc[0], 0.0)
    credit_history = sample.get("Credit_History", pd.Series([np.nan])).iloc[0]

    total_income = max(applicant_income + coapplicant_income, 1.0)
    loan_to_income = loan_amount / total_income

    if pd.notna(credit_history) and float(credit_history) < 1:
        risk += 0.25
        risk_notes.append({"feature": "Credit_History", "impact": "Low credit history increases risk."})
    if loan_to_income > 0.25:
        risk += 0.2
        risk_notes.append({"feature": "LoanAmount", "impact": "Loan amount is high relative to income."})
    if applicant_income < 3000:
        risk += 0.15
        risk_notes.append({"feature": "ApplicantIncome", "impact": "Low applicant income raises risk."})
    if coapplicant_income <= 0:
        risk += 0.05
        risk_notes.append({"feature": "CoapplicantIncome", "impact": "No coapplicant support increases risk."})
    if "Self_Employed" in sample.columns and str(sample["Self_Employed"].iloc[0]).strip().lower() == "yes":
        risk += 0.05
        risk_notes.append({"feature": "Self_Employed", "impact": "Self-employment can add income volatility."})
    if "Dependents" in sample.columns:
        dependents = str(sample["Dependents"].iloc[0]).strip()
        if dependents in {"3", "3+"}:
            risk += 0.05
            risk_notes.append({"feature": "Dependents", "impact": "Higher number of dependents can increase burden."})

    return float(np.clip(risk, 0.0, 1.0)), risk_notes


def explain_prediction(
    bundle: ModelBundle,
    sample: pd.DataFrame,
    top_k: int = 5,
) -> dict[str, Any]:
    transformed_sample = bundle.preprocessor.transform(sample)
    model_estimator = get_model_estimator(bundle)
    feature_names = get_feature_names(bundle)

    explanation: dict[str, Any] = {
        "shap_available": False,
        "lime_available": False,
        "top_features": [],
    }

    # Fallback explanation that always works.
    fallback_risk, fallback_notes = _heuristic_risk(bundle, sample)
    explanation["heuristic_risk"] = fallback_risk
    explanation["risk_notes"] = fallback_notes

    if shap is not None:
        try:  # pragma: no cover - optional dependency
            if hasattr(model_estimator, "feature_importances_"):
                explainer = shap.TreeExplainer(model_estimator)
                shap_values = explainer.shap_values(transformed_sample)
            elif hasattr(model_estimator, "coef_"):
                reference = bundle.reference_data if bundle.reference_data is not None else sample
                background = bundle.preprocessor.transform(reference)
                explainer = shap.LinearExplainer(model_estimator, background, feature_perturbation="interventional")
                shap_values = explainer.shap_values(transformed_sample)
            else:
                explainer = shap.Explainer(model_estimator, transformed_sample)
                shap_values = explainer(transformed_sample).values

            values = shap_values[0] if isinstance(shap_values, list) else shap_values[0]
            ranked = sorted(
                zip(feature_names, np.asarray(values).tolist()),
                key=lambda item: abs(item[1]),
                reverse=True,
            )
            explanation["top_features"] = _format_explanation_items(ranked[:top_k])
            explanation["shap_available"] = True
        except Exception:
            explanation["top_features"] = fallback_notes[:top_k]
    else:
        explanation["top_features"] = fallback_notes[:top_k]

    try:  # pragma: no cover - optional dependency
        from lime.lime_tabular import LimeTabularExplainer  # type: ignore

        reference = bundle.reference_data if bundle.reference_data is not None else sample
        lime_training = bundle.preprocessor.transform(reference)
        lime_feature_names = get_feature_names(bundle)
        class_names = [bundle.negative_label, bundle.positive_label]

        explainer = LimeTabularExplainer(
            training_data=lime_training,
            feature_names=lime_feature_names,
            class_names=class_names,
            mode="classification",
        )
        lime_exp = explainer.explain_instance(
            transformed_sample[0],
            lambda values: model_estimator.predict_proba(values),
            num_features=top_k,
        )
        explanation["lime_available"] = True
        explanation["lime_features"] = _format_explanation_items(lime_exp.as_list())
    except Exception:
        pass

    return explanation


def predict_application(bundle: ModelBundle, sample_payload: dict[str, Any]) -> dict[str, Any]:
    sample = _coerce_sample(bundle, sample_payload)
    model = bundle.model

    prediction = int(model.predict(sample)[0])
    probabilities = model.predict_proba(sample)[0] if hasattr(model, "predict_proba") else None
    approval_probability = float(probabilities[1]) if probabilities is not None else None
    rejection_probability = float(probabilities[0]) if probabilities is not None else None

    fraud_models = bundle.fraud_models or {}
    transformed = bundle.preprocessor.transform(sample)
    iso = fraud_models.get("isolation_forest")
    lof = fraud_models.get("local_outlier_factor")
    stats = fraud_models.get("stats", {})

    anomaly_scores = {}
    if iso is not None:
        iso_raw = float(-iso.decision_function(transformed)[0])
        anomaly_scores["isolation_forest"] = _normalize_risk(
            iso_raw,
            stats.get("iso_mean", iso_raw),
            stats.get("iso_std", 1.0),
        )
    if lof is not None:
        lof_raw = float(-lof.decision_function(transformed)[0])
        anomaly_scores["local_outlier_factor"] = _normalize_risk(
            lof_raw,
            stats.get("lof_mean", lof_raw),
            stats.get("lof_std", 1.0),
        )

    heuristic_risk, risk_notes = _heuristic_risk(bundle, sample)
    model_risk = 1.0 - approval_probability if approval_probability is not None else 0.5
    anomaly_risk = float(np.mean(list(anomaly_scores.values()))) if anomaly_scores else 0.0
    combined_risk = float(np.clip(0.6 * model_risk + 0.25 * anomaly_risk + 0.15 * heuristic_risk, 0.0, 1.0))

    explanation = explain_prediction(bundle, sample)
    if not explanation.get("top_features"):
        explanation["top_features"] = risk_notes

    result = {
        "prediction": prediction,
        "label": bundle.positive_label if prediction == 1 else bundle.negative_label,
        "approval_probability": approval_probability,
        "rejection_probability": rejection_probability,
        "risk_score": round(combined_risk * 100, 2),
        "fraud_flag": combined_risk >= 0.7 or any(score >= 0.75 for score in anomaly_scores.values()),
        "anomaly_scores": anomaly_scores,
        "heuristic_risk": round(heuristic_risk * 100, 2),
        "top_features": explanation.get("top_features", []),
        "lime_features": explanation.get("lime_features", []),
        "risk_notes": explanation.get("risk_notes", []),
        "shap_available": explanation.get("shap_available", False),
        "lime_available": explanation.get("lime_available", False),
        "raw_input": sample_payload,
    }
    return result


def save_metrics_summary(metrics_df: pd.DataFrame, output_dir: str = "outputs") -> str:
    summary = metrics_df.sort_values(by="f1_score", ascending=False).to_dict(orient="records")
    path = Path(output_dir) / "metrics_summary.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return str(path)
