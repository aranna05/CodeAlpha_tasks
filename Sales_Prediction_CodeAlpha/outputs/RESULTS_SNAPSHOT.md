# SALES SIGNAL | Results snapshot

## Selected model
Polynomial Regression (degree 2)

| Metric | Holdout result |
|---|---:|
| MAE | 0.485 |
| RMSE | 0.621 |
| R² | 0.988 |
| 5-fold training CV MAE | 0.441 |

## Interpretation
The model's mean absolute error is about 0.48 sales units on the held-out sample. R² of 0.988 means it explains about 98.8% of the variation in this particular holdout split. These are dataset-specific evaluation results, not a guarantee of future performance.

## Predictive signal
  Feature  MAE increase when shuffled  Variation across shuffles
       TV                    4.477206                   0.523010
    Radio                    2.647134                   0.260558
Newspaper                   -0.003214                   0.018479

## Business guardrails
- TV has the strongest simple correlation with sales in the supplied data; this is association, not causation.
- Permutation importance reflects the selected model's reliance on features, not incremental ROI.
- No dates exist, so seasonality, time-based forecasting and COVID impact cannot be analyzed.
- No segment or platform columns exist, so segment/platform effects cannot be estimated.
- Run controlled experiments before treating model insights as budget-allocation evidence.
