"""
SALES SIGNAL | Advertising-to-sales prediction pipeline
Run: python sales_prediction.py
Requirements: pandas, numpy, matplotlib, scikit-learn, joblib
"""
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "Advertising.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

# 1. Load and validate
raw = pd.read_csv(DATA)
data = raw.drop(columns=["Unnamed: 0"], errors="ignore").copy()
required = ["TV", "Radio", "Newspaper", "Sales"]
missing_cols = sorted(set(required) - set(data.columns))
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")
data = data[required].copy()
for col in required:
    data[col] = pd.to_numeric(data[col], errors="coerce")
if data.isna().any().any():
    raise ValueError("Missing/non-numeric values detected; review before modeling.")
if data.duplicated().any():
    print(f"Note: {data.duplicated().sum()} duplicate rows found.")
if (data[["TV", "Radio", "Newspaper"]] < 0).any().any():
    raise ValueError("Advertising spend cannot be negative.")

X, y = data[["TV", "Radio", "Newspaper"]], data["Sales"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# 2. Compare interpretable and non-linear approaches
models = {
    "Linear Regression": Pipeline([
        ("scaler", StandardScaler()), ("model", LinearRegression())
    ]),
    "Ridge Regression": Pipeline([
        ("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))
    ]),
    "Polynomial Regression (degree 2)": Pipeline([
        ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
        ("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))
    ]),
    "Random Forest": RandomForestRegressor(
        n_estimators=500, min_samples_leaf=2, max_features=1.0,
        random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.03, max_depth=2,
        loss="huber", random_state=42
    )
}

results, fitted = [], {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    cv_mae = -cross_val_score(
        model, X_train, y_train, cv=cv, scoring="neg_mean_absolute_error"
    ).mean()
    results.append({
        "Model": name, "CV MAE (5-fold)": cv_mae,
        "Test MAE": mean_absolute_error(y_test, pred),
        "Test RMSE": mean_squared_error(y_test, pred) ** 0.5,
        "Test R²": r2_score(y_test, pred)
    })
    fitted[name] = model

leaderboard = pd.DataFrame(results).sort_values("CV MAE (5-fold)")
leaderboard.to_csv(OUT / "model_comparison.csv", index=False)
best_name = leaderboard.iloc[0]["Model"]
best_model = fitted[best_name]
pred = best_model.predict(X_test)

print("\\nSALES SIGNAL | Model evaluation")
print(leaderboard.to_string(index=False))
print(f"\\nSelected model: {best_name}")
print(f"Holdout MAE:  {mean_absolute_error(y_test, pred):.3f}")
print(f"Holdout RMSE: {mean_squared_error(y_test, pred)**0.5:.3f}")
print(f"Holdout R²:   {r2_score(y_test, pred):.3f}")

pd.DataFrame({
    "Actual Sales": y_test.values, "Predicted Sales": pred,
    "Prediction Error": y_test.values-pred
}).to_csv(OUT / "test_predictions.csv", index=False)

# 3. Model explanation: permutation importance on unseen test rows
perm = permutation_importance(
    best_model, X_test, y_test, n_repeats=30,
    random_state=42, scoring="neg_mean_absolute_error"
)
importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance (MAE increase)": perm.importances_mean,
    "Std": perm.importances_std
}).sort_values("Importance (MAE increase)", ascending=False)
importance.to_csv(OUT / "permutation_importance.csv", index=False)
print("\\nPermutation importance (higher = more predictive signal):")
print(importance.to_string(index=False))

# 4. Save reusable model and a cautious prediction helper
joblib.dump(best_model, OUT / "sales_prediction_model.joblib")
print("\\nSaved model to outputs/sales_prediction_model.joblib")
print("Reminder: predictions are estimates, not guaranteed sales outcomes.")
