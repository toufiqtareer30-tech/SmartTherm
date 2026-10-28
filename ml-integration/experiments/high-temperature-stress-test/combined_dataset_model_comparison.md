# Combined-data model comparison

All rows within each table use the same examples. MSE is RMSE squared and is reported in °C².

## Held-out synthetic session: five-minute target above 90°C

| Model | Examples | MAE (°C) | MSE (°C²) | RMSE (°C) |
|---|---:|---:|---:|---:|
| Persistence | 207 | 2.4998 | 17.6879 | 4.2057 |
| Installed residual XGBoost (real-only) | 207 | 35.0263 | 1252.9688 | 35.3973 |
| Direct XGBoost (train split) | 207 | 1.2733 | 2.6105 | 1.6157 |
| Gradient Boosting (train split) | 207 | 1.3692 | 3.0846 | 1.7563 |
| Residual XGBoost (synthetic-aware train split) | 207 | 1.3253 | 2.7869 | 1.6694 |

## Held-out synthetic session: all five-minute targets

| Model | Examples | MAE (°C) | MSE (°C²) | RMSE (°C) |
|---|---:|---:|---:|---:|
| Persistence | 302 | 4.0822 | 30.9948 | 5.5673 |
| Installed residual XGBoost (real-only) | 302 | 31.96 | 1065.4479 | 32.6412 |
| Direct XGBoost (train split) | 302 | 1.5909 | 4.2845 | 2.0699 |
| Gradient Boosting (train split) | 302 | 1.7678 | 5.4536 | 2.3353 |
| Residual XGBoost (synthetic-aware train split) | 302 | 1.5529 | 4.2957 | 2.0726 |

## Entire combined dataset: in-sample fit

| Model | Examples | MAE (°C) | MSE (°C²) | RMSE (°C) |
|---|---:|---:|---:|---:|
| Persistence | 12016 | 7.102 | 140.2235 | 11.8416 |
| Installed residual XGBoost (real-only) | 12016 | 10.7695 | 279.6587 | 16.723 |
| Direct XGBoost (all-data fit) | 12016 | 4.1944 | 46.7459 | 6.8371 |
| Gradient Boosting (all-data fit) | 12016 | 4.4861 | 70.2127 | 8.3793 |
| Residual XGBoost (all-data fit) | 12016 | 3.4248 | 30.868 | 5.5559 |

## Entire combined dataset: in-sample fit, target above 90°C

| Model | Examples | MAE (°C) | MSE (°C²) | RMSE (°C) |
|---|---:|---:|---:|---:|
| Persistence | 2008 | 7.6823 | 220.0743 | 14.8349 |
| Installed residual XGBoost (real-only) | 2008 | 34.2595 | 1202.4458 | 34.6763 |
| Direct XGBoost (all-data fit) | 2008 | 5.4093 | 118.3396 | 10.8784 |
| Gradient Boosting (all-data fit) | 2008 | 7.0001 | 206.1838 | 14.3591 |
| Residual XGBoost (all-data fit) | 2008 | 4.3633 | 73.6027 | 8.5792 |

The held-out session was not used to train the synthetic-aware models or choose their residual correction. The entire-dataset tables are in-sample because those models were fit on all 7,181 rows; they are descriptive fit errors, not unbiased generalization scores.

Synthetic temperatures and workloads are not real-hardware accuracy evidence. The experimental models are not installed in live inference.
