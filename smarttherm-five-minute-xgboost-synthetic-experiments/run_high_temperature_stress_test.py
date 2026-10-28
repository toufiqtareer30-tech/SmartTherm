from __future__ import annotations

import csv
import json
import math
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[3]
ML_DIR = ROOT / "ml-integration"
sys.path.insert(0, str(ML_DIR))

from smarttherm_ml.dataset import build_examples, discover_csv_files, load_points
from smarttherm_ml.features import FEATURE_NAMES
from smarttherm_ml.training import (
    _features_matrix,
    _metrics,
    _select_residual_correction_weight,
    split_chronologically,
    train_component,
)


OUTPUT_DIR = Path(__file__).resolve().parent
CSV_PATH = OUTPUT_DIR / "mixed_cpu_high_temperature_stress.csv"
REPORT_PATH = OUTPUT_DIR / "mixed_cpu_high_temperature_report.json"
EXPERIMENTAL_MODEL_PATH = OUTPUT_DIR / "experimental_full_dataset_cpu_xgboostresidual.joblib"
COMPARISON_CSV_PATH = OUTPUT_DIR / "combined_dataset_model_comparison.csv"
COMPARISON_SUMMARY_PATH = OUTPUT_DIR / "combined_dataset_model_comparison.md"
MODEL_DIR = ML_DIR / "models"
FIELDS = [
    "schemaVersion",
    "sessionId",
    "timestamp",
    "cpuTemperatureC",
    "gpuTemperatureC",
    "cpuUtilizationPercent",
    "cpuClockMhz",
    "cpuPowerW",
    "cpuFanSpeedRpm",
    "gpuUtilizationPercent",
    "memoryUtilizationPercent",
    "gpuSensor",
]


def _noise(session: int, step: int, salt: int) -> float:
    # Deterministic bounded variation keeps the CSV reproducible.
    return 0.32 * math.sin(step * 1.71 + session * 0.83 + salt)


def generate_csv() -> dict[str, int]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    start = datetime(2026, 10, 27, tzinfo=timezone.utc)
    total_rows = 0
    cpu_over_90 = 0
    normal_rows = 0

    with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        # Eight sustained/high-load sessions plus four lower-load sessions.
        # Every session is 30 minutes at the collector's five-second cadence.
        for session in range(12):
            high_load = session < 8
            session_id = f"synthetic-stress-{session + 1:02d}"
            session_start = start + timedelta(hours=2 * session)
            for step in range(361):
                elapsed = step * 5
                if high_load:
                    if elapsed <= 300:
                        temperature = 82.0 + 10.0 * elapsed / 300.0
                    elif elapsed <= 1200:
                        temperature = 94.0 + 1.45 * math.sin(elapsed / 47.0 + session)
                    else:
                        temperature = 94.0 - 0.028 * (elapsed - 1200)
                    temperature += _noise(session, step, 0)
                    cpu_load = 91.0 + 7.0 * math.sin(elapsed / 31.0 + session * 0.4)
                    cpu_load = min(100.0, max(80.0, cpu_load))
                    clock = 3900.0 + 240.0 * math.sin(elapsed / 53.0 + session)
                    power = 105.0 + 18.0 * math.sin(elapsed / 37.0 + session * 0.2)
                    fan = 3600.0 + 650.0 * math.sin(elapsed / 61.0 + session)
                    gpu_load = 15.0 + 8.0 * math.sin(elapsed / 43.0 + session)
                else:
                    if elapsed <= 600:
                        temperature = 48.0 + 19.0 * elapsed / 600.0
                    elif elapsed <= 1200:
                        temperature = 67.0 + 5.0 * math.sin(elapsed / 83.0 + session)
                    else:
                        temperature = 67.0 - 0.018 * (elapsed - 1200)
                    temperature += _noise(session, step, 2)
                    cpu_load = 27.0 + 17.0 * math.sin(elapsed / 57.0 + session)
                    cpu_load = min(52.0, max(8.0, cpu_load))
                    clock = 2200.0 + 420.0 * math.sin(elapsed / 72.0 + session)
                    power = 28.0 + 12.0 * math.sin(elapsed / 58.0 + session)
                    fan = 1500.0 + 500.0 * math.sin(elapsed / 95.0 + session)
                    gpu_load = 8.0 + 6.0 * math.sin(elapsed / 69.0 + session)

                cpu_temperature = round(min(99.0, max(35.0, temperature)), 2)
                cpu_load = round(min(100.0, max(0.0, cpu_load)), 2)
                gpu_load = round(min(100.0, max(0.0, gpu_load)), 2)
                if cpu_temperature > 90.0:
                    cpu_over_90 += 1
                else:
                    normal_rows += 1

                timestamp = session_start + timedelta(seconds=elapsed)
                writer.writerow({
                    "schemaVersion": 1,
                    "sessionId": session_id,
                    "timestamp": timestamp.isoformat().replace("+00:00", "Z"),
                    "cpuTemperatureC": cpu_temperature,
                    "gpuTemperatureC": round(48.0 + gpu_load * 0.19 + _noise(session, step, 3), 2),
                    "cpuUtilizationPercent": cpu_load,
                    "cpuClockMhz": round(max(800.0, clock), 2),
                    "cpuPowerW": round(max(5.0, power), 2),
                    "cpuFanSpeedRpm": round(max(600.0, fan), 2),
                    "gpuUtilizationPercent": round(gpu_load, 2),
                    "memoryUtilizationPercent": round(42.0 + 5.0 * math.sin(elapsed / 109.0 + session), 2),
                    "gpuSensor": "SYNTHETIC_STRESS_TEST",
                })
                total_rows += 1

    return {
        "rows": total_rows,
        "cpuReadingsOver90C": cpu_over_90,
        "cpuReadingsAtOrBelow90C": normal_rows,
    }


def evaluate_installed_model() -> dict[str, object]:
    manifest = json.loads((MODEL_DIR / "manifest.json").read_text(encoding="utf-8"))
    entry = manifest.get("models", {}).get("CPU")
    if not isinstance(entry, dict):
        raise RuntimeError("No installed CPU model is registered.")
    artifact_path = MODEL_DIR / entry["artifact"]
    artifact = joblib.load(artifact_path)
    if artifact.get("targetHorizonSeconds") != 300:
        raise RuntimeError("The installed CPU model is not a five-minute model.")

    points, rejected_rows = load_points([CSV_PATH])
    examples_by_component, rejected_targets = build_examples(points)
    examples = examples_by_component["CPU"]
    if not examples:
        raise RuntimeError("The stress-test CSV produced no five-minute CPU examples.")

    features = _features_matrix(examples)
    actual = np.asarray([item.target_c for item in examples], dtype=float)
    persistence = np.asarray([item.persistence_prediction_c for item in examples], dtype=float)
    predicted = np.asarray(artifact["pipeline"].predict(features), dtype=float)

    def scored(mask: np.ndarray) -> dict[str, object]:
        if not np.any(mask):
            return {"examples": 0, "model": None, "persistence": None}
        return {
            "examples": int(np.sum(mask)),
            "model": _metrics(actual[mask], predicted[mask]),
            "persistence": _metrics(actual[mask], persistence[mask]),
        }

    current = features[:, FEATURE_NAMES.index("temperature_current_c")]
    return {
        "evaluationType": "synthetic_external_stress_test_only",
        "model": entry.get("model"),
        "modelTrainedAt": entry.get("trainedAt"),
        "targetHorizonSeconds": 300,
        "sourceCsv": str(CSV_PATH),
        "rejectedRows": dict(rejected_rows),
        "rejectedTargets": dict(rejected_targets),
        "allFiveMinuteExamples": scored(np.ones(len(examples), dtype=bool)),
        "targetAbove90C": scored(actual > 90.0),
        "currentAndTargetAbove90C": scored((current > 90.0) & (actual > 90.0)),
        "note": (
            "Temperatures and workloads in this CSV are synthetic. These errors describe "
            "only this generated stress-test distribution; they are not real-hardware accuracy. "
            "This file is under ml-integration/experiments/high-temperature-stress-test and is excluded from automatic training discovery."
        ),
    }


def evaluate_isolated_augmented_training() -> dict[str, object]:
    real_files = discover_csv_files(ROOT)
    combined_points, rejected_rows = load_points([*real_files, CSV_PATH])
    examples_by_component, rejected_targets = build_examples(combined_points)
    examples = examples_by_component["CPU"]
    _, _, test = split_chronologically(examples)
    test_hot = sum(item.target_c > 90.0 for item in test)
    test_hot_stay_hot = sum(
        item.target_c > 90.0
        and (item.features["temperature_current_c"] or 0.0) > 90.0
        for item in test
    )

    # Use a disposable model directory so this experiment cannot replace the
    # installed live model or its manifest.
    with tempfile.TemporaryDirectory(
        prefix=".experiment-models-", dir=OUTPUT_DIR
    ) as temporary:
        training = train_component("CPU", examples, Path(temporary))

    return {
        "evaluationType": "isolated_real_plus_synthetic_training_experiment",
        "realCsvFiles": len(real_files),
        "realAndSyntheticSensorRows": len(combined_points),
        "rejectedRows": dict(rejected_rows),
        "rejectedTargets": dict(rejected_targets),
        "fiveMinuteExamples": len(examples),
        "heldOutTargetAbove90CExamples": test_hot,
        "heldOutCurrentAndTargetAbove90CExamples": test_hot_stay_hot,
        "trainingResult": training,
        "note": (
            "The temporary experimental artifact was not installed. Promotion status and metrics "
            "come from the existing chronological validation/holdout gates; the installed model and "
            "manifest are unchanged. Synthetic examples are not real-hardware evidence."
        ),
    }


def evaluate_synthetic_aware_xgboost() -> dict[str, object]:
    real_files = discover_csv_files(ROOT)
    points, _ = load_points([*real_files, CSV_PATH])
    examples_by_component, _ = build_examples(points)
    examples = examples_by_component["CPU"]

    # Keep full sessions separate: six hot sessions plus normal synthetic and
    # all real examples for training, one hot session for validation, and a
    # different hot session for the final stress-test holdout.
    train_ids = {f"synthetic-stress-{index:02d}" for index in range(1, 7)}
    validation_id = "synthetic-stress-07"
    test_id = "synthetic-stress-08"
    train = [
        item for item in examples
        if not item.session_id.startswith("synthetic-stress-")
        or item.session_id in train_ids
        or item.session_id in {f"synthetic-stress-{index:02d}" for index in range(9, 13)}
    ]
    validation = [item for item in examples if item.session_id == validation_id]
    test = [item for item in examples if item.session_id == test_id]
    if not train or not validation or not test:
        raise RuntimeError("Could not form separate training, validation, and hot-session test sets.")

    x_train = _features_matrix(train)
    x_validation = _features_matrix(validation)
    x_test = _features_matrix(test)
    current_index = FEATURE_NAMES.index("temperature_current_c")
    y_train = np.asarray([item.target_c for item in train], dtype=float)
    y_validation = np.asarray([item.target_c for item in validation], dtype=float)
    y_test = np.asarray([item.target_c for item in test], dtype=float)
    current_validation = x_validation[:, current_index]
    current_test = x_test[:, current_index]

    experiments: list[dict[str, object]] = []
    for depth, min_child_weight in ((3, 2), (4, 2), (5, 5)):
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("regressor", XGBRegressor(
                objective="reg:squarederror",
                n_estimators=300,
                max_depth=depth,
                learning_rate=0.04,
                min_child_weight=min_child_weight,
                subsample=0.9,
                colsample_bytree=0.95,
                reg_lambda=1.0,
                tree_method="hist",
                n_jobs=2,
                random_state=42,
            )),
        ])
        pipeline.fit(x_train, y_train - x_train[:, current_index])
        validation_residual = pipeline.predict(x_validation)
        weight = _select_residual_correction_weight(
            y_validation,
            current_validation,
            validation_residual,
        )
        validation_prediction = current_validation + weight * validation_residual
        validation_metrics = _metrics(y_validation, validation_prediction)
        val_baseline = _metrics(y_validation, current_validation)
        experiments.append({
            "depth": depth,
            "minChildWeight": min_child_weight,
            "pipeline": pipeline,
            "correctionWeight": weight,
            "validation": validation_metrics,
            "validationBaseline": val_baseline,
            "validationScore": (
                validation_metrics["mae_c"] / max(val_baseline["mae_c"], 1e-12)
                + validation_metrics["rmse_c"] / max(val_baseline["rmse_c"], 1e-12)
            ),
        })

    selected = min(experiments, key=lambda item: item["validationScore"])
    selected_pipeline = selected["pipeline"]
    test_residual = selected_pipeline.predict(x_test)
    test_prediction = current_test + float(selected["correctionWeight"]) * test_residual
    target_hot = y_test > 90.0
    return {
        "trainingExamples": len(train),
        "validationExamples": len(validation),
        "heldOutHotSessionExamples": len(test),
        "heldOutTargetAbove90CExamples": int(np.sum(target_hot)),
        "validationSession": validation_id,
        "heldOutSession": test_id,
        "selectedHyperparameters": {
            "maxDepth": selected["depth"],
            "minChildWeight": selected["minChildWeight"],
            "correctionWeight": selected["correctionWeight"],
        },
        "validation": {
            "model": selected["validation"],
            "persistence": selected["validationBaseline"],
        },
        "heldOutAll": {
            "model": _metrics(y_test, test_prediction),
            "persistence": _metrics(y_test, current_test),
        },
        "heldOutTargetAbove90C": {
            "model": _metrics(y_test[target_hot], test_prediction[target_hot]) if np.any(target_hot) else None,
            "persistence": _metrics(y_test[target_hot], current_test[target_hot]) if np.any(target_hot) else None,
        },
        "beatsPersistenceOnBothHighTempMetrics": bool(
            np.any(target_hot)
            and _metrics(y_test[target_hot], test_prediction[target_hot])["mae_c"]
                < _metrics(y_test[target_hot], current_test[target_hot])["mae_c"]
            and _metrics(y_test[target_hot], test_prediction[target_hot])["rmse_c"]
                < _metrics(y_test[target_hot], current_test[target_hot])["rmse_c"]
        ),
        "note": (
            "This is a separate synthetic-aware experiment. The test session is held out from training "
            "and correction-weight selection. The candidate is not installed or used by live inference."
        ),
    }


def fit_experimental_model_on_all_data() -> dict[str, object]:
    real_files = discover_csv_files(ROOT)
    points, rejected_rows = load_points([*real_files, CSV_PATH])
    examples_by_component, rejected_targets = build_examples(points)
    examples = examples_by_component["CPU"]
    features = _features_matrix(examples)
    actual = np.asarray([item.target_c for item in examples], dtype=float)
    persistence = np.asarray([item.persistence_prediction_c for item in examples], dtype=float)
    current_index = FEATURE_NAMES.index("temperature_current_c")
    residual_target = actual - features[:, current_index]

    # Hyperparameters and correction weight were selected using a separate
    # synthetic validation session in evaluate_synthetic_aware_xgboost().
    correction_weight = 1.0
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("regressor", XGBRegressor(
            objective="reg:squarederror",
            n_estimators=300,
            max_depth=5,
            learning_rate=0.04,
            min_child_weight=5,
            subsample=0.9,
            colsample_bytree=0.95,
            reg_lambda=1.0,
            tree_method="hist",
            n_jobs=2,
            random_state=42,
        )),
    ])
    pipeline.fit(features, residual_target)
    predicted = features[:, current_index] + correction_weight * pipeline.predict(features)
    synthetic_mask = np.asarray(
        [item.session_id.startswith("synthetic-stress-") for item in examples],
        dtype=bool,
    )
    real_mask = ~synthetic_mask
    synthetic_hot_mask = synthetic_mask & (actual > 90.0)
    artifact = {
        "schemaVersion": 2,
        "targetHorizonSeconds": 300,
        "component": "CPU",
        "model": "ExperimentalXGBoostResidualSyntheticAugmented",
        "trainedAt": datetime.now(timezone.utc).isoformat(),
        "trainingExamples": len(examples),
        "fitScope": "all_real_history_plus_mixed_synthetic_stress_test",
        "correctionWeight": correction_weight,
        "featureNames": FEATURE_NAMES,
        "pipeline": pipeline,
        "experimentalOnly": True,
    }
    joblib.dump(artifact, EXPERIMENTAL_MODEL_PATH)

    def scores(mask: np.ndarray) -> dict[str, object]:
        return {
            "examples": int(np.sum(mask)),
            "model": _metrics(actual[mask], predicted[mask]) if np.any(mask) else None,
            "persistence": _metrics(actual[mask], persistence[mask]) if np.any(mask) else None,
        }

    return {
        "realCsvFiles": len(real_files),
        "realSensorRows": sum(not p.session_id.startswith("synthetic-stress-") for p in points),
        "syntheticSensorRows": sum(p.session_id.startswith("synthetic-stress-") for p in points),
        "rejectedRows": dict(rejected_rows),
        "rejectedTargets": dict(rejected_targets),
        "fitExamples": len(examples),
        "modelArtifact": str(EXPERIMENTAL_MODEL_PATH),
        "selectedHyperparametersFromValidation": {
            "maxDepth": 5,
            "minChildWeight": 5,
            "correctionWeight": correction_weight,
        },
        "inSampleMetrics_not_holdout": {
            "allExamples": scores(np.ones(len(examples), dtype=bool)),
            "realExamples": scores(real_mask),
            "syntheticExamples": scores(synthetic_mask),
            "syntheticTargetsAbove90C": scores(synthetic_hot_mask),
        },
        "note": (
            "This experimental model was fit on every listed example, so these in-sample metrics "
            "are not unbiased performance estimates. Use the separate syntheticAwareXGBoostExperiment "
            "held-out session scores for the independent test result. This artifact is not registered "
            "or loaded by SmartTherm."
        ),
    }


def compare_all_models_on_common_data() -> dict[str, object]:
    real_files = discover_csv_files(ROOT)
    points, _ = load_points([*real_files, CSV_PATH])
    examples_by_component, _ = build_examples(points)
    examples = examples_by_component["CPU"]
    hot_train_ids = {f"synthetic-stress-{index:02d}" for index in range(1, 7)}
    normal_train_ids = {f"synthetic-stress-{index:02d}" for index in range(9, 13)}
    train = [
        item for item in examples
        if not item.session_id.startswith("synthetic-stress-")
        or item.session_id in hot_train_ids
        or item.session_id in normal_train_ids
    ]
    validation = [item for item in examples if item.session_id == "synthetic-stress-07"]
    test = [item for item in examples if item.session_id == "synthetic-stress-08"]
    x_train, x_validation, x_test = (
        _features_matrix(part) for part in (train, validation, test)
    )
    x_all = _features_matrix(examples)
    current_index = FEATURE_NAMES.index("temperature_current_c")
    current_train = x_train[:, current_index]
    current_validation = x_validation[:, current_index]
    current_test = x_test[:, current_index]
    current_all = x_all[:, current_index]
    y_train = np.asarray([item.target_c for item in train], dtype=float)
    y_validation = np.asarray([item.target_c for item in validation], dtype=float)
    y_test = np.asarray([item.target_c for item in test], dtype=float)
    y_all = np.asarray([item.target_c for item in examples], dtype=float)

    def direct_xgb() -> Pipeline:
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("regressor", XGBRegressor(
                objective="reg:squarederror", n_estimators=250, max_depth=4,
                learning_rate=0.04, min_child_weight=5, subsample=0.85,
                colsample_bytree=0.9, reg_lambda=2.0, tree_method="hist",
                n_jobs=2, random_state=42)),
        ])

    def gradient_boosting() -> Pipeline:
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("regressor", GradientBoostingRegressor(
                loss="huber", n_estimators=150, learning_rate=0.04,
                max_depth=3, min_samples_leaf=5, random_state=42)),
        ])

    direct = direct_xgb().fit(x_train, y_train)
    gradient = gradient_boosting().fit(x_train, y_train)
    residual_candidates = []
    for depth, min_child_weight in ((3, 2), (4, 2), (5, 5)):
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("regressor", XGBRegressor(
                objective="reg:squarederror", n_estimators=300,
                max_depth=depth, learning_rate=0.04,
                min_child_weight=min_child_weight, subsample=0.9,
                colsample_bytree=0.95, reg_lambda=1.0, tree_method="hist",
                n_jobs=2, random_state=42)),
        ])
        model.fit(x_train, y_train - current_train)
        residual_validation = model.predict(x_validation)
        weight = _select_residual_correction_weight(
            y_validation, current_validation, residual_validation
        )
        val_pred = current_validation + weight * residual_validation
        val_model_metrics = _metrics(y_validation, val_pred)
        val_base_metrics = _metrics(y_validation, current_validation)
        residual_candidates.append({
            "model": model,
            "weight": weight,
            "depth": depth,
            "minChildWeight": min_child_weight,
            "score": (
                val_model_metrics["mae_c"] / max(val_base_metrics["mae_c"], 1e-12)
                + val_model_metrics["rmse_c"] / max(val_base_metrics["rmse_c"], 1e-12)
            ),
        })
    selected = min(residual_candidates, key=lambda candidate: candidate["score"])
    residual_test = current_test + float(selected["weight"]) * selected["model"].predict(x_test)

    manifest = json.loads((MODEL_DIR / "manifest.json").read_text(encoding="utf-8"))
    active_entry = manifest["models"]["CPU"]
    active_artifact = joblib.load(MODEL_DIR / active_entry["artifact"])
    active_test = np.asarray(active_artifact["pipeline"].predict(x_test), dtype=float)
    heldout_predictions = {
        "Persistence": current_test,
        "Installed residual XGBoost (real-only)": active_test,
        "Direct XGBoost (train split)": np.asarray(direct.predict(x_test)),
        "Gradient Boosting (train split)": np.asarray(gradient.predict(x_test)),
        "Residual XGBoost (synthetic-aware train split)": residual_test,
    }

    # Refit each trainable algorithm on every combined example for an in-sample
    # comparison, clearly labeled as such in the output table.
    direct_all = direct_xgb().fit(x_all, y_all)
    gradient_all = gradient_boosting().fit(x_all, y_all)
    residual_all_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("regressor", XGBRegressor(
            objective="reg:squarederror", n_estimators=300,
            max_depth=int(selected["depth"]), learning_rate=0.04,
            min_child_weight=int(selected["minChildWeight"]), subsample=0.9,
            colsample_bytree=0.95, reg_lambda=1.0, tree_method="hist",
            n_jobs=2, random_state=42)),
    ]).fit(x_all, y_all - current_all)
    all_predictions = {
        "Persistence": current_all,
        "Installed residual XGBoost (real-only)": np.asarray(active_artifact["pipeline"].predict(x_all)),
        "Direct XGBoost (all-data fit)": np.asarray(direct_all.predict(x_all)),
        "Gradient Boosting (all-data fit)": np.asarray(gradient_all.predict(x_all)),
        "Residual XGBoost (all-data fit)": current_all + float(selected["weight"]) * residual_all_model.predict(x_all),
    }

    rows: list[dict[str, object]] = []

    def add_rows(scope: str, actual: np.ndarray, prediction_map: dict[str, np.ndarray]) -> None:
        masks = {
            "all": np.ones(len(actual), dtype=bool),
            "target_above_90C": actual > 90.0,
        }
        for subset, mask in masks.items():
            for model_name, values in prediction_map.items():
                metrics = _metrics(actual[mask], values[mask])
                rows.append({
                    "evaluation_scope": scope,
                    "subset": subset,
                    "model": model_name,
                    "examples": int(np.sum(mask)),
                    "mae_c": metrics["mae_c"],
                    "mse_c2": round(metrics["rmse_c"] ** 2, 4),
                    "rmse_c": metrics["rmse_c"],
                })

    add_rows("heldout_synthetic_session_08", y_test, heldout_predictions)
    add_rows("combined_real_plus_synthetic_in_sample", y_all, all_predictions)
    with COMPARISON_CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = [
        "# Combined-data model comparison",
        "",
        "All rows within each table use the same examples. MSE is RMSE squared and is reported in °C².",
    ]
    scopes = [
        (
            "heldout_synthetic_session_08",
            "target_above_90C",
            "Held-out synthetic session: five-minute target above 90°C",
        ),
        (
            "heldout_synthetic_session_08",
            "all",
            "Held-out synthetic session: all five-minute targets",
        ),
        (
            "combined_real_plus_synthetic_in_sample",
            "all",
            "Entire combined dataset: in-sample fit",
        ),
        (
            "combined_real_plus_synthetic_in_sample",
            "target_above_90C",
            "Entire combined dataset: in-sample fit, target above 90°C",
        ),
    ]
    for scope, subset, title in scopes:
        selected_rows = [
            row for row in rows
            if row["evaluation_scope"] == scope and row["subset"] == subset
        ]
        summary.extend([
            "",
            f"## {title}",
            "",
            "| Model | Examples | MAE (°C) | MSE (°C²) | RMSE (°C) |",
            "|---|---:|---:|---:|---:|",
        ])
        for row in selected_rows:
            summary.append(
                f"| {row['model']} | {row['examples']} | {row['mae_c']} | "
                f"{row['mse_c2']} | {row['rmse_c']} |"
            )
    summary.extend([
        "",
        "The held-out session was not used to train the synthetic-aware models or choose their residual correction. The entire-dataset tables are in-sample because those models were fit on all 7,181 rows; they are descriptive fit errors, not unbiased generalization scores.",
        "",
        "Synthetic temperatures and workloads are not real-hardware accuracy evidence. The experimental models are not installed in live inference.",
        "",
    ])
    COMPARISON_SUMMARY_PATH.write_text("\n".join(summary), encoding="utf-8")

    return {
        "comparisonCsv": str(COMPARISON_CSV_PATH),
        "comparisonSummary": str(COMPARISON_SUMMARY_PATH),
        "commonHeldOutSession": "synthetic-stress-08",
        "commonHeldOutExamples": len(test),
        "heldOutTargetAbove90CExamples": int(np.sum(y_test > 90.0)),
        "fullCombinedExamples": len(examples),
        "selectedResidualHyperparameters": {
            "maxDepth": selected["depth"],
            "minChildWeight": selected["minChildWeight"],
            "correctionWeight": selected["weight"],
        },
        "rows": rows,
        "note": (
            "The held-out synthetic session is not used for training or model selection. "
            "The combined_real_plus_synthetic_in_sample scope trains on the same rows it scores, "
            "so those values are descriptive fit errors, not generalization estimates."
        ),
    }
def main() -> None:
    counts = generate_csv()
    report = evaluate_installed_model()
    report["csvCounts"] = counts
    report["augmentedTrainingExperiment"] = evaluate_isolated_augmented_training()
    report["syntheticAwareXGBoostExperiment"] = evaluate_synthetic_aware_xgboost()
    report["fullDatasetExperimentalFit"] = fit_experimental_model_on_all_data()
    report["commonDataModelComparison"] = compare_all_models_on_common_data()
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
