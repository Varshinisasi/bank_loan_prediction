from __future__ import annotations

from pathlib import Path

from flask import Flask, flash, redirect, render_template, request, url_for

from loan_system import load_bundle, predict_application, train_project


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_PATH = OUTPUT_DIR / "best_model_bundle.joblib"
DEFAULT_DATA_PATH = BASE_DIR / "data" / "raw" / "loan_dataset_raw.csv"


FIELD_SPECS = [
    {
        "name": "Gender",
        "label": "Gender",
        "type": "select",
        "default": "Male",
        "options": ["Male", "Female"],
    },
    {
        "name": "Married",
        "label": "Married",
        "type": "select",
        "default": "Yes",
        "options": ["Yes", "No"],
    },
    {
        "name": "Dependents",
        "label": "Dependents",
        "type": "select",
        "default": "0",
        "options": ["0", "1", "2", "3+"],
    },
    {
        "name": "Education",
        "label": "Education",
        "type": "select",
        "default": "Graduate",
        "options": ["Graduate", "Not Graduate"],
    },
    {
        "name": "Self_Employed",
        "label": "Self Employed",
        "type": "select",
        "default": "No",
        "options": ["No", "Yes"],
    },
    {"name": "ApplicantIncome", "label": "Applicant Income", "type": "number", "default": 4583},
    {"name": "CoapplicantIncome", "label": "Coapplicant Income", "type": "number", "default": 150},
    {"name": "LoanAmount", "label": "Loan Amount", "type": "number", "default": 128},
    {"name": "Loan_Amount_Term", "label": "Loan Amount Term", "type": "number", "default": 360},
    {
        "name": "Credit_History",
        "label": "Credit History",
        "type": "select",
        "default": 1,
        "options": [1, 0],
        "help": "1 = good/available credit history, 0 = poor or unavailable credit history",
    },
    {
        "name": "Property_Area",
        "label": "Property Area",
        "type": "select",
        "default": "Rural",
        "options": ["Urban", "Semiurban", "Rural"],
    },
]


def ensure_bundle():
    data_path = DEFAULT_DATA_PATH if DEFAULT_DATA_PATH.exists() else None
    if MODEL_PATH.exists():
        if data_path is None:
            return load_bundle(str(MODEL_PATH))
        if MODEL_PATH.stat().st_mtime >= data_path.stat().st_mtime:
            return load_bundle(str(MODEL_PATH))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bundle, metrics = train_project(
        data_path=str(data_path) if data_path is not None else None,
        target_column=None,
        output_dir=str(OUTPUT_DIR),
        data_dir=str(BASE_DIR / "data"),
        test_size=0.2,
        random_state=42,
    )
    metrics.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    return bundle


app = Flask(__name__)
app.secret_key = "loan-risk-dashboard-secret"


def default_form_values():
    return {spec["name"]: spec["default"] for spec in FIELD_SPECS}


@app.route("/", methods=["GET", "POST"])
def index():
    bundle = ensure_bundle()
    result = None
    active_panel = request.args.get("panel", "home")
    form_values = default_form_values()
    prediction_message = None

    if request.method == "POST":
        for spec in FIELD_SPECS:
            form_values[spec["name"]] = request.form.get(spec["name"], spec["default"])

        try:
            result = predict_application(bundle, form_values)
            prediction_message = "Prediction completed. You can view the results in Prediction Summary and Top Explanation Factors."
            active_panel = "summary"
        except Exception as exc:
            flash(f"Could not generate prediction: {exc}", "error")

    return render_template(
        "index.html",
        fields=FIELD_SPECS,
        result=result,
        form_values=form_values,
        active_panel=active_panel,
        prediction_message=prediction_message,
        model_name=bundle.model_name,
        training_rows=bundle.training_rows,
    )


@app.route("/train", methods=["POST"])
def train_model():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data_path = request.form.get("data_path") or str(DEFAULT_DATA_PATH if DEFAULT_DATA_PATH.exists() else "")
    bundle, metrics = train_project(
        data_path=data_path or None,
        target_column=request.form.get("target_column") or None,
        output_dir=str(OUTPUT_DIR),
        data_dir=str(BASE_DIR / "data"),
        test_size=0.2,
        random_state=42,
    )
    metrics.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    flash(f"Training complete. Best model: {bundle.model_name}", "success")
    return redirect(url_for("index", panel="summary"))


if __name__ == "__main__":
    app.run(debug=True)
