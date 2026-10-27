# SmartTherm ML training and inference

## Current model status

The active integration serves a validated **five-minute CPU forecast**. GPU remains explicitly unconfigured; the existing ten-minute model artifacts and pipeline are preserved in `ml-integration\models\` and are not loaded or overwritten. The new schema-v2 model directory is separate at `ml-integration\models-five-minute\`. Check `GET http://127.0.0.1:8090/health` for the loaded model name, version, horizon, validation status, and training state. No untrained trend estimate is substituted.

## Collect real training data

Prerequisites: 64-bit Python 3.12+ and the .NET 10 SDK. Starting the Spring backend automatically starts the collector. For collection without the dashboard/API, it can also be run directly:

```powershell
dotnet run --project .\ThermalCollector\ThermalCollector.csproj
```

Each collector run now appends a versioned CSV such as:

```text
ThermalCollector\data\training_telemetry_<session-id>_part000.csv
```

The file has a UTC timestamp and stable columns for session ID, CPU/GPU temperature, utilization, CPU clock, CPU power, CPU fan speed, memory load, and GPU sensor identity. Sensor fields unavailable on your hardware are blank, never filled with fabricated values. A collector restart begins a new session. Files roll at about 10 MiB; old training files are pruned toward a 512 MiB total retention budget.

CPU temperature data is hardware- and driver-dependent. Check the collector's sensor diagnostics and leave HWiNFO running if its local shared-memory readings are required. Sampling is about every five seconds. Collect across real workloads and idle periods, and avoid stopping the collector during useful sessions. The five-minute CPU model needs continuous same-session samples and an actual reading about 300 seconds after each feature timestamp; collection time alone is not sufficient when sensor metrics are missing or large gaps split a session.

## Set up Python and assess data

In PowerShell from the BTP folder:

```powershell
python -m venv .\ml-integration\.venv
& .\ml-integration\.venv\Scripts\python.exe -m pip install -r .\ml-integration\requirements.txt
& .\ml-integration\.venv\Scripts\python.exe .\ml-integration\assess_data.py
```

For a machine-readable report:

```powershell
& .\ml-integration\.venv\Scripts\python.exe .\ml-integration\assess_data.py --json
```

The assessor reports usable five-minute pairs and eligible chronological train, validation, and test counts. It normalizes collector CSV column names but does not join separate CSV sessions, invent missing temperatures, or interpolate across gaps. A label requires an actual later CPU observation approximately 300 seconds after each feature timestamp (within 15 seconds), continuous same-session samples with at most 20 seconds between readings, valid temperature bounds, and enough history for features. GPU pairs are assessed for data coverage only; this integration does not train or serve a GPU model.

## Train and evaluate

### Synthetic development data (not live model data)

The legacy synthetic development generator remains available only for isolated software experiments:

```powershell
& .\ml-integration\.venv\Scripts\python.exe .\ml-integration\generate_synthetic.py
```

Generated synthetic data is not used by the live model pipeline, is not a forecast source, and must not be described as measured hardware behavior.

The existing experiment command checks generated-data schema and basic statistics, then writes research artifacts under `synthetic-lab\artifacts\`. Those files are intentionally outside both model directories and are never loaded by the application. Synthetic metrics test software behavior only; the live CPU model is trained exclusively from real collector telemetry.

The local inference service checks changed training data at startup and every 15 minutes by default. It runs training in a background worker, independent of five-second sensor collection and prediction. A persisted fingerprint prevents repeatedly training on unchanged CSV files. Training uses only real, same-session CPU targets and chronological partitions; the five-minute candidate must improve on persistence in validation and held-out MAE/RMSE and must not regress an active five-minute model. Promotion writes a versioned artifact and manifest in `models-five-minute`; the registry reloads the manifest on its next health or prediction request without restarting the service. Missing/invalid manifests fail closed. Insufficient data, rejected candidates, and training errors are reported through `GET /health` under `training`; no prediction is fabricated.

Automatic training can be disabled with `SMARTTHERM_AUTO_TRAINING=false`. Set `SMARTTHERM_TRAINING_CHECK_INTERVAL_SECONDS` to change the interval (minimum 60 seconds). Only one training job may use a model directory at a time, including a manual CLI run. The five-minute model directory is intentionally separate from the preserved ten-minute `models` directory. Do not point `SMARTTHERM_MODEL_DIR` at the latter: training refuses to promote over schema-v1 ten-minute artifacts.

To assess labels manually, run:

```powershell
& .\ml-integration\.venv\Scripts\python.exe .\ml-integration\assess_data.py
```

The CPU training pipeline compares direct XGBoost, residual XGBoost, and scikit-learn Gradient Boosting against persistence and a five-minute linear-trend baseline. It uses the 15 ordered features in `smarttherm_ml\features.py`, median imputers, chronological one-hour session blocks, and a five-minute embargo between train/validation/test boundaries. It requires at least 300 training examples and 50 validation plus 50 held-out test examples after splitting. A candidate is saved only if it improves on persistence on both MAE and RMSE in validation and held-out test data, and does not regress an active five-minute model. Reports include both baseline metrics; otherwise they say `NOT_PROMOTED` or `INSUFFICIENT_DATA`. Do not lower evidence thresholds or promote an unvalidated model.

An optional manual run uses the same safeguards:

```powershell
& .\ml-integration\.venv\Scripts\python.exe .\ml-integration\train_models.py
```

Accepted five-minute CPU models and `manifest.json` are stored locally in `ml-integration\models-five-minute\`, ignored by Git. This command honors `SMARTTHERM_MODEL_DIR`; its default is the separate five-minute model directory. Training does not run per reading or block live collection/inference. Models use `joblib`, so load only locally generated/trusted artifacts. The service reloads changed manifests on health and prediction requests and fails closed if a new manifest or artifact is invalid.

## Start the local inference service

The all-services launcher starts the Python service in a separate PowerShell window. To run it on its own:

```powershell
.\scripts\start-ml-service.ps1
```

The script creates `ml-integration\.venv` if needed, installs the pinned dependencies, then starts Uvicorn on the host/port from `SMARTTHERM_ML_URL` (default **127.0.0.1:8090**). Model artifacts are loaded from `SMARTTHERM_MODEL_DIR` (default `ml-integration\models-five-minute\`). The root `.\scripts\start-smarttherm.ps1` command starts Python inference, waits for `/health`, and only then starts the Java API/collector and frontend. No cloud, public-network listener, database, or Docker is used.

Endpoints:

- `GET /health`: reports API health, per-component availability, model name/version, horizon, validation status, error reason, and automatic-training status.
- `POST /training/check`: queues an immediate background check after the dashboard's **Start calibration** action. It does not bypass the real-data or model-promotion gates.
- `POST /predict`: accepts up to 150 chronologically ordered, timezone-stamped real readings from the latest 300 seconds. A ready forecast includes CPU temperature in °C, UTC generated/target timestamps, `horizonMinutes: 5`, model name, and model version. Out-of-range, stale, missing, or gapped inputs are rejected or returned with null predictions and an explicit unavailable reason.

The Java API maintains this active 300-second feature window separately from historical training CSVs, calls `/predict` with bounded timeouts/retries, and validates response freshness, component readings, model metadata, and temperature range. The existing frontend contract remains additive: each forecast now carries its horizon and model version. If Python or a model is unavailable, predicted temperatures remain null; this does not prevent independent sensor collection. Existing warning rules are not changed by the five-minute integration.

## Data contract and evaluation

- Model schema version: `2`; timestamps are UTC; temperatures are Celsius; clock speed is MHz; load is percent.
- CPU target: actual CPU temperature sampled 300 seconds later ±15 seconds in the same continuous session. Gaps over 20 seconds break the segment.
- CPU-only model: `GPU` returns `NOT_CONFIGURED`; this does not remove any preserved ten-minute artifacts or change sensor collection.
- Ordered feature fields: `temperature_current_c`, `temperature_mean_30s_c`, `temperature_mean_60s_c`, `temperature_mean_300s_c`, `temperature_min_60s_c`, `temperature_max_60s_c`, `temperature_slope_30s_c_per_min`, `temperature_slope_300s_c_per_min`, `cpu_utilization_percent`, `cpu_utilization_mean_60s_percent`, `cpu_clock_mhz`, `cpu_clock_mean_60s_mhz`, `gpu_utilization_percent`, `gpu_utilization_mean_60s_percent`, `memory_utilization_percent`.
- Missing features are handled by the saved training pipeline; all reported metrics are computed against actual later readings, not synthetic labels.
- Reports include sample counts and per-component MAE/RMSE; these are not a guarantee of future accuracy.

The Python tests are synthetic unit fixtures for schema, target, time-split, and API behavior; they are not training data or product-performance evidence. Actual hardware/model efficacy remains unverified until the collector records sufficient real telemetry and a candidate passes the held-out test.
