# SmartTherm five-minute XGBoost integration bundle

This package contains reference snapshots for the five-minute residual-XGBoost integration, the current CPU model artifact, and its manifest. The project version used to build these files may differ from the recipient's project. Review and merge the changes into the recipient's corresponding files; do not blindly overwrite files from another version.

## Runtime changes

- The prediction horizon, target labels, inference lookback, backend active window, and dashboard forecast display use five minutes.
- Training compares direct XGBoost, residual XGBoost, and Gradient Boosting. Residual correction weight is selected on the validation partition.
- Candidate promotion still requires beating persistence on validation and untouched holdout metrics. After promotion checks pass, the installed model is refit on all eligible collected examples.
- Model artifacts use schema version 2 and target horizon 300 seconds. The supplied manifest registers a CPU model only. GPU candidates did not pass validation and were not promoted.
- Background training is enabled by default in the service and checks for changed telemetry on its configured interval (15 minutes by default).

## Supplied model

The `ml-integration/models` folder contains the promoted CPU artifact and its manifest. The model was fit on 3,473 eligible five-minute CPU examples collected through 2026-10-27 local time. It was trained with Python 3.13.7, NumPy 2.4.3, scikit-learn 1.9.1, and XGBoost 3.4.1. Joblib model files are Python-version and library-version sensitive; if the recipient's runtime differs or loading fails, install compatible dependencies or train a new local model from their collected telemetry.

## Files

The Java backend and Python service files implement the horizon and model contract. `frontend/src/App.jsx` updates the forecast demo and display. The included tests are reference updates for those contracts. `ml-integration/smarttherm_ml/features.py` is included as the model feature-contract dependency.

The bundle does not include collected sensor CSVs or other workspace changes.
