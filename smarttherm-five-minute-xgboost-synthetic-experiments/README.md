# ML experiments

This folder holds offline research material. The automatic trainer scans the project telemetry locations, not this folder, so synthetic examples here are never treated as live collector data.

## Contents

- `high-temperature-stress-test/` contains the mixed synthetic CPU stress CSV, the experiment runner, the model comparison CSV and JSON report, and an experimental model fit on all available real plus synthetic examples. The synthetic set contains 4,332 readings, including 1,762 above 90°C and 2,570 at or below 90°C.
- `reports/` contains outputs from the research notebook and train-only augmentation experiment.
- `synthetic-lab/` contains separate synthetic development data and its evaluation artifacts.
- `SmartTherm_Hybrid_Model_Experiment.ipynb`, `augment_high_temp_experiment.py`, and `compare_feature_scaling.py` are offline research tools.

The comparison CSV reports both a held-out synthetic session and full combined-data in-sample scores. Only the held-out results estimate performance on unseen sessions; the all-data fit metrics are descriptive because those same rows were used to fit the experimental models. Synthetic metrics are not real-hardware accuracy and do not promote or replace the live model.
