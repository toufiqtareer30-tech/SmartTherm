"""Compare scaling and causal temperature features on shared 2-6 minute test rows.

Run from the repository root with the notebook dependencies installed:
    python ml-integration/experiments/compare_feature_scaling.py
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[2]
ML_DIR = ROOT / "ml-integration"
sys.path.insert(0, str(ML_DIR))
import smarttherm_ml.dataset as dataset  # noqa: E402
from smarttherm_ml.features import FEATURE_NAMES  # noqa: E402

NOTEBOOK = ML_DIR / "experiments" / "SmartTherm_Hybrid_Model_Experiment.ipynb"
HORIZONS = range(2, 7)
VARIANTS = ("baseline", "standard_scaled", "minmax_scaled", "previous_row_delta", "delta_and_previous_2m_max_ratio")


def load_clean_continuous_points():
    """Reuse the notebook's exact source discovery, deduplication, and continuity setup."""
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    namespace = {"__file__": str(NOTEBOOK)}
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        exec("".join(notebook["cells"][1]["source"]), namespace)
        exec("".join(notebook["cells"][3]["source"]), namespace)
    return namespace["continuous_points"]


def make_models(variant: str, adaptive: bool = False):
    scaler = {"standard_scaled": StandardScaler(), "minmax_scaled": MinMaxScaler()}.get(variant)
    def pipeline(regressor):
        steps = [("imputer", SimpleImputer(strategy="median", keep_empty_features=True))]
        if scaler is not None:
            steps.append(("scaler", scaler))
        steps.append(("regressor", regressor))
        return Pipeline(steps)

    if adaptive:
        return {"Adaptive Gradient Boosting": pipeline(GradientBoostingRegressor(
            loss="huber", n_estimators=150, learning_rate=0.04,
            max_depth=3, min_samples_leaf=5, random_state=42))}
    return {
        "Fixed Gradient Boosting": pipeline(GradientBoostingRegressor(
            loss="huber", n_estimators=150, learning_rate=0.04,
            max_depth=3, min_samples_leaf=5, random_state=42)),
        "Fixed XGBoost": pipeline(XGBRegressor(
            objective="reg:squarederror", n_estimators=250, max_depth=4,
            learning_rate=0.04, min_child_weight=5, subsample=0.85,
            colsample_bytree=0.9, reg_lambda=2.0, tree_method="hist",
            n_jobs=2, random_state=42)),
    }


def augment_features(examples, previous_values, variant):
    matrix = np.asarray([
        [np.nan if example.features.get(name) is None else float(example.features[name]) for name in FEATURE_NAMES]
        for example in examples
    ], dtype=float)
    if variant not in {"previous_row_delta", "delta_and_previous_2m_max_ratio"}:
        return matrix
    extra = []
    for example in examples:
        previous_temp, previous_max = previous_values[(example.session_id, example.timestamp)]
        current = float(example.features["temperature_current_c"])
        delta = current - previous_temp if previous_temp is not None else np.nan
        ratio = current / previous_max if previous_max is not None and previous_max != 0 else np.nan
        extra.append([delta] if variant == "previous_row_delta" else [delta, ratio])
    return np.column_stack([matrix, np.asarray(extra, dtype=float)])


def metrics(actual, prediction):
    err = prediction - actual
    return {
        "MAE_C": mean_absolute_error(actual, prediction),
        "RMSE_C": mean_squared_error(actual, prediction) ** 0.5,
        "R2": r2_score(actual, prediction),
        "mean_signed_error_C": float(np.mean(err)),
    }


def main():
    points = load_clean_continuous_points()
    grouped = defaultdict(list)
    for point in sorted(points, key=lambda item: (item.session_id, item.timestamp)):
        if point.cpu_temperature_c is not None:
            grouped[point.session_id].append(point)
    prior = {}
    for session_id, rows in grouped.items():
        for i, point in enumerate(rows):
            window = [r.cpu_temperature_c for r in rows[:i]
                      if 0 < (point.timestamp - r.timestamp).total_seconds() <= 120]
            previous = rows[i - 1].cpu_temperature_c if i > 0 else None
            prior[(session_id, point.timestamp)] = (previous, max(window) if window else None)

    examples_by_horizon = {}
    original_horizon = dataset.TARGET_HORIZON_SECONDS
    try:
        for horizon in HORIZONS:
            dataset.TARGET_HORIZON_SECONDS = horizon * 60
            examples, _ = dataset.build_examples(points)
            examples_by_horizon[horizon] = examples["CPU"]
    finally:
        dataset.TARGET_HORIZON_SECONDS = original_horizon

    # Use only anchors having a target at every requested horizon, so each horizon
    # and variant is evaluated on the same timestamp population.
    anchor_sets = [
        {(e.session_id, e.timestamp) for e in examples_by_horizon[h]}
        for h in HORIZONS
    ]
    common = set.intersection(*anchor_sets)
    ordered = sorted(common, key=lambda k: k[1])
    n = len(ordered)
    if n < 100:
        raise RuntimeError(f"Expected a substantial common evaluation set, found {n} anchors")
    # Match the existing sweep's 1,396 / 370 / 551 chronological partition.
    train_end = 1396
    validation_end = 1766
    train_keys = set(ordered[:train_end])
    validation_keys = set(ordered[train_end:validation_end])
    test_keys = set(ordered[validation_end:])

    results = []
    for variant in VARIANTS:
        for horizon in HORIZONS:
            by_key = {(e.session_id, e.timestamp): e for e in examples_by_horizon[horizon]}
            train = [by_key[k] for k in ordered[:train_end]]
            test = [by_key[k] for k in ordered[validation_end:]]
            x_train = augment_features(train, prior, variant)
            x_test = augment_features(test, prior, variant)
            y_train = np.asarray([e.target_c for e in train], dtype=float)
            y_test = np.asarray([e.target_c for e in test], dtype=float)
            for name, model in make_models(variant).items():
                model.fit(x_train, y_train)
                prediction = model.predict(x_test)
                results.append({"variant": variant, "model": name, "horizon_min": horizon,
                                "train_rows": len(train), "test_examples": len(test),
                                **metrics(y_test, prediction)})

        # Adaptive GB training includes 2-10 minute labels from the shared train anchors.
        adapt_train_x, adapt_train_y = [], []
        for horizon in range(2, 11):
            dataset.TARGET_HORIZON_SECONDS = horizon * 60
            examples, _ = dataset.build_examples(points)
            for example in examples["CPU"]:
                key = (example.session_id, example.timestamp)
                if key in train_keys:
                    adapt_train_x.append((horizon, example))
                    adapt_train_y.append(example.target_c)
        dataset.TARGET_HORIZON_SECONDS = original_horizon
        # Fit one model per variant, test at each common anchor and requested horizon.
        adapt_matrix = lambda items: np.column_stack([
            augment_features([e for _, e in items], prior, variant),
            np.asarray([h for h, _ in items], dtype=float),
        ])
        adaptive_model = make_models(variant, adaptive=True)["Adaptive Gradient Boosting"]
        adaptive_model.fit(adapt_matrix(adapt_train_x), np.asarray(adapt_train_y, dtype=float))
        for horizon in HORIZONS:
            test = [next(e for e in examples_by_horizon[horizon]
                         if (e.session_id, e.timestamp) == key) for key in ordered[validation_end:]]
            x_test = np.column_stack([
                augment_features(test, prior, variant),
                np.full(len(test), horizon, dtype=float),
            ])
            y_test = np.asarray([e.target_c for e in test], dtype=float)
            prediction = adaptive_model.predict(x_test)
            results.append({"variant": variant, "model": "Adaptive Gradient Boosting",
                            "horizon_min": horizon, "train_rows": len(adapt_train_y),
                            "test_examples": len(test), **metrics(y_test, prediction)})

    frame = pd.DataFrame(results)
    output = ML_DIR / "experiments" / "reports" / "scaling_and_temperature_feature_metrics_2_to_6.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"Shared anchors: {n}; train/validation/test={len(train_keys)}/{len(validation_keys)}/{len(test_keys)}")
    print(f"Saved: {output}")
    best = frame.loc[frame.groupby(["model", "variant"])["MAE_C"].idxmin()]
    print("Best horizon per model and variant:")
    print(best.sort_values(["model", "MAE_C"])[["model", "variant", "horizon_min", "MAE_C", "RMSE_C", "R2"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("Baseline reproduction check:")
    print(frame[(frame.variant == "baseline") & (frame.horizon_min == 5)]
          [["model", "MAE_C", "RMSE_C", "R2"]].to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
