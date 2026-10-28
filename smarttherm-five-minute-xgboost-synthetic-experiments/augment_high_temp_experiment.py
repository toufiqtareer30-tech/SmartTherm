"""Create and evaluate a train-only synthetic augmentation for 5-minute hot targets.

Generated examples are written to experiments/reports/ and never to collector telemetry files.
Run from the repository root:
    python ml-integration/experiments/augment_high_temp_experiment.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_score, recall_score, f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[2]
ML_DIR = ROOT / "ml-integration"
sys.path.insert(0, str(ML_DIR))
from smarttherm_ml.dataset import build_examples, discover_csv_files, load_points  # noqa: E402
from smarttherm_ml.features import FEATURE_NAMES  # noqa: E402
from smarttherm_ml.training import split_chronologically, _select_residual_correction_weight  # noqa: E402

THRESHOLD_C = 90.0
SYNTHETIC_ROWS = 500
RANDOM_SEED = 20261028


def matrix(examples):
    return np.asarray([
        [np.nan if example.features.get(name) is None else float(example.features[name])
         for name in FEATURE_NAMES]
        for example in examples
    ], dtype=float)


def fit_residual_model(x_train, y_train, x_validation, y_validation):
    current_index = FEATURE_NAMES.index("temperature_current_c")
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("regressor", XGBRegressor(
            objective="reg:squarederror", n_estimators=250, max_depth=4,
            learning_rate=0.04, min_child_weight=5, subsample=0.85,
            colsample_bytree=0.9, reg_lambda=2.0, tree_method="hist",
            n_jobs=2, random_state=42)),
    ])
    current_train = x_train[:, current_index]
    residual_train = y_train - current_train
    pipe.fit(x_train, residual_train)
    validation_residual = pipe.predict(x_validation)
    weight = _select_residual_correction_weight(
        y_validation,
        x_validation[:, current_index],
        validation_residual,
    )
    return pipe, weight


def score(actual, predicted, current):
    err = predicted - actual
    truth_hot = actual >= THRESHOLD_C
    pred_hot = predicted >= THRESHOLD_C
    high_current = current >= THRESHOLD_C
    return {
        "examples": int(len(actual)),
        "MAE_C": float(mean_absolute_error(actual, predicted)),
        "RMSE_C": float(mean_squared_error(actual, predicted) ** 0.5),
        "predicted_hot_count": int(pred_hot.sum()),
        "actual_hot_count": int(truth_hot.sum()),
        "hot_precision": float(precision_score(truth_hot, pred_hot, zero_division=0)),
        "hot_recall": float(recall_score(truth_hot, pred_hot, zero_division=0)),
        "hot_f1": float(f1_score(truth_hot, pred_hot, zero_division=0)),
        "current_ge_90_examples": int(high_current.sum()),
        "false_hot_when_current_ge_90": int((high_current & pred_hot & ~truth_hot).sum()),
        "true_hot_when_current_ge_90": int((high_current & pred_hot & truth_hot).sum()),
        "MAE_when_current_ge_90_C": (
            float(mean_absolute_error(actual[high_current], predicted[high_current]))
            if high_current.any() else None
        ),
        "mean_signed_error_C": float(np.mean(err)),
    }


def main():
    files = discover_csv_files(ROOT)
    points, rejected = load_points(files)
    examples_by_component, _ = build_examples(points)
    examples = examples_by_component["CPU"]
    train, validation, test = split_chronologically(examples)
    hot_train = [item for item in train if item.target_c >= THRESHOLD_C]
    if len(hot_train) < 2:
        raise RuntimeError("Need at least two training examples with a 5-minute target >= 90 C.")

    rng = np.random.default_rng(RANDOM_SEED)
    x_train = matrix(train)
    y_train = np.asarray([item.target_c for item in train], dtype=float)
    x_validation = matrix(validation)
    y_validation = np.asarray([item.target_c for item in validation], dtype=float)
    x_test = matrix(test)
    y_test = np.asarray([item.target_c for item in test], dtype=float)
    current_idx = FEATURE_NAMES.index("temperature_current_c")

    # Impute and scale only the real training subset used to locate neighbors.
    neighbor_imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    hot_x_raw = matrix(hot_train)
    hot_x = neighbor_imputer.fit_transform(hot_x_raw)
    neighbor_scaler = StandardScaler().fit(hot_x)
    hot_z = neighbor_scaler.transform(hot_x)
    neighbor_count = min(6, len(hot_train))
    neighbors = NearestNeighbors(n_neighbors=neighbor_count).fit(hot_z)
    neighbor_ids = neighbors.kneighbors(return_distance=False)

    generated_x = np.empty((SYNTHETIC_ROWS, len(FEATURE_NAMES)), dtype=float)
    generated_y = np.empty(SYNTHETIC_ROWS, dtype=float)
    parent_left = np.empty(SYNTHETIC_ROWS, dtype=int)
    parent_right = np.empty(SYNTHETIC_ROWS, dtype=int)
    alpha_values = np.empty(SYNTHETIC_ROWS, dtype=float)
    # Interpolate only between nearby real training examples whose future target
    # is >=90 C. The untouched validation and test partitions are never sampled.
    for i in range(SYNTHETIC_ROWS):
        left = int(rng.integers(len(hot_train)))
        candidates = neighbor_ids[left][neighbor_ids[left] != left]
        right = int(rng.choice(candidates))
        alpha = float(rng.uniform(0.15, 0.85))
        generated_x[i] = hot_x[left] + alpha * (hot_x[right] - hot_x[left])
        generated_y[i] = hot_train[left].target_c + alpha * (hot_train[right].target_c - hot_train[left].target_c)
        parent_left[i], parent_right[i], alpha_values[i] = left, right, alpha

    # Validation chooses the residual correction weight. Test remains untouched.
    real_model, real_weight = fit_residual_model(x_train, y_train, x_validation, y_validation)
    aug_x_train = np.vstack([x_train, generated_x])
    aug_y_train = np.concatenate([y_train, generated_y])
    aug_model, aug_weight = fit_residual_model(aug_x_train, aug_y_train, x_validation, y_validation)

    real_prediction = x_test[:, current_idx] + real_weight * real_model.predict(x_test)
    aug_prediction = x_test[:, current_idx] + aug_weight * aug_model.predict(x_test)
    persistence_prediction = x_test[:, current_idx]
    results = []
    for name, prediction, weight in [
        ("Persistence baseline", persistence_prediction, 0.0),
        ("Real training rows only", real_prediction, real_weight),
        ("Real + 500 synthetic hot-target rows", aug_prediction, aug_weight),
    ]:
        results.append({"model_variant": name, "correction_weight": weight,
                        **score(y_test, prediction, x_test[:, current_idx])})

    report_dir = ML_DIR / "experiments" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    feature_rows = pd.DataFrame(generated_x, columns=FEATURE_NAMES)
    feature_rows["target_temperature_5m_c"] = generated_y
    feature_rows["is_synthetic"] = True
    feature_rows["generation_method"] = "nearest-neighbor interpolation among train-only examples with target >=90C"
    feature_rows["parent_train_example_a"] = parent_left
    feature_rows["parent_train_example_b"] = parent_right
    feature_rows["interpolation_alpha"] = alpha_values
    feature_path = report_dir / "synthetic_high_temp_examples_500.csv"
    feature_rows.to_csv(feature_path, index=False)
    results_path = report_dir / "high_temp_augmentation_test_metrics.json"
    results_path.write_text(json.dumps({
        "threshold_c": THRESHOLD_C,
        "synthetic_rows": SYNTHETIC_ROWS,
        "generation_method": "Nearest-neighbor interpolation between positive examples from the chronological training split only.",
        "dataset": {"eligible_examples": len(examples), "train": len(train),
                    "validation": len(validation), "test": len(test),
                    "train_hot_targets": len(hot_train), "synthetic_rows_added_to_fit": SYNTHETIC_ROWS},
        "metrics": results,
        "data_note": "Synthetic examples are for this experiment only; no collector CSV, saved model, or active manifest was modified.",
    }, indent=2), encoding="utf-8")

    print(f"Eligible CPU examples: {len(examples)}; split train/validation/test: {len(train)}/{len(validation)}/{len(test)}")
    print(f"Training examples with 5-minute target >=90 C: {len(hot_train)}; synthetic examples generated: {SYNTHETIC_ROWS}")
    print(f"Saved synthetic examples: {feature_path}")
    print(f"Saved holdout comparison: {results_path}")
    print(pd.DataFrame(results).to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
