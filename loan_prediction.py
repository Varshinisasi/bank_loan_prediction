from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

from loan_system import load_bundle, predict_application, train_project


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Loan risk prediction and training CLI")
    parser.add_argument("--data", default=None, help="Path to the training dataset")
    parser.add_argument("--target", default=None, help="Target column name")
    parser.add_argument("--output-dir", default="outputs", help="Folder for saved artifacts")
    parser.add_argument("--data-dir", default="data", help="Folder for raw and split CSV files")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split size")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--predict-json", default=None, help="JSON string for a single applicant")
    parser.add_argument("--predict-file", default=None, help="Path to a JSON file for prediction")
    parser.add_argument(
        "--model-path",
        default="outputs/best_model_bundle.joblib",
        help="Path to the saved model bundle",
    )
    return parser.parse_args()


def load_payload(args: argparse.Namespace):
    if args.predict_file:
        payload_path = Path(args.predict_file)
        return json.loads(payload_path.read_text(encoding="utf-8"))
    if args.predict_json:
        try:
            return json.loads(args.predict_json)
        except json.JSONDecodeError:
            payload = ast.literal_eval(args.predict_json)
            if not isinstance(payload, dict):
                raise ValueError("Prediction input must be a JSON object.")
            return payload
    return None


def main() -> None:
    args = parse_args()
    payload = load_payload(args)

    if payload is not None:
        bundle = load_bundle(args.model_path)
        result = predict_application(bundle, payload)
        print(json.dumps(result, indent=2))
        return

    bundle, metrics = train_project(
        data_path=args.data,
        target_column=args.target,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
        test_size=args.test_size,
        random_state=args.random_state,
    )
    print(f"Best model: {bundle.model_name}")
    print(metrics.to_string(index=False))
    print(f"Saved model bundle to: {Path(args.output_dir) / 'best_model_bundle.joblib'}")


if __name__ == "__main__":
    main()
