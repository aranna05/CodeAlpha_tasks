# ✦ SALES SIGNAL — Sales Prediction with Python

An end-to-end, reproducible regression project using the supplied `Advertising.csv`.

## What is included
- `Sales_Signal_Sales_Prediction.ipynb` — polished, step-by-step notebook with EDA, cleaning, model comparison, diagnostics, interpretation and scenario prediction.
- `sales_prediction.py` — standalone training/evaluation pipeline.
- `data/Advertising.csv` — source dataset extracted from the supplied ZIP.
- `outputs/` — model comparison, holdout predictions, permutation importance, charts and trained model.

## Dataset facts
- 200 rows and 4 analytical columns: TV, Radio, Newspaper, Sales.
- The uploaded file also contained `Unnamed: 0`, treated as an exported row index and excluded from modeling.
- No missing values were found in the supplied data.
- No date, target-segment or platform field exists. Therefore this is a regression project, not time-series forecasting or segment/platform analysis.
- Sales units and spend currency are not documented in the file, so output values retain the dataset's native units.

## Run it
```bash
pip install pandas numpy matplotlib scikit-learn joblib jupyter
jupyter notebook Sales_Signal_Sales_Prediction.ipynb
```
Or run the standalone pipeline from this folder:
```bash
python sales_prediction.py
```

## Model selection
Candidates: Linear Regression, Ridge, degree-2 Polynomial Regression, Random Forest and Gradient Boosting. A fixed 80/20 holdout split and shuffled 5-fold CV are used. The notebook's model pipelines keep transformations inside each fold.

## Important interpretation note
Predictive importance is not causal return on investment. Use controlled experiments or stronger causal designs before reallocating real budgets. For true future forecasting, add date-indexed sales and advertising spend, then validate with chronological backtesting.
