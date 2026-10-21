"""Regression pipeline for Track 1 (Chocolate Bar Ratings / Flavors of Cacao).

All 10 required algorithms:
  1. Linear Regression (Baseline; interpret coefficients)
  2. Ridge Regression (L2 regularization; tune alpha)
  3. Lasso Regression (L1 regularization; feature sparsity)
  4. ElasticNet Regression (Combined L1+L2; tune l1_ratio)
  5. Polynomial Regression (degree 2 PolynomialFeatures + LinearRegression)
  6. Decision Tree Regressor (Tune max_depth; feature importance)
  7. Random Forest Regressor (Ensemble baseline; tune n_estimators)
  8. Gradient Boosting Regressor (GBM; tune learning_rate)
  9. Support Vector Regressor (SVR) (Scale features first; tune C, kernel)
 10. K-Nearest Neighbors Regressor (Tune k; scaling impact)
"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score, GridSearchCV

from . import config as C
from . import data_loader as dl


def get_baseline_models():
    """Return dictionary of the 10 algorithms matching assignment requirements."""
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=C.RANDOM_STATE),
        "Lasso Regression": Lasso(alpha=0.01, random_state=C.RANDOM_STATE),
        "ElasticNet Regression": ElasticNet(l1_ratio=0.5, random_state=C.RANDOM_STATE),
        "Polynomial Regression": Pipeline([
            ("poly", PolynomialFeatures(degree=2)),
            ("linear", LinearRegression())
        ]),
        "Decision Tree Regressor": DecisionTreeRegressor(max_depth=5, random_state=C.RANDOM_STATE),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=C.RANDOM_STATE),
        "Gradient Boosting Regressor": GradientBoostingRegressor(learning_rate=0.1, random_state=C.RANDOM_STATE),
        "Support Vector Regressor": SVR(C=1.0, kernel="rbf"),
        "K-Nearest Neighbors Regressor": KNeighborsRegressor(n_neighbors=5),
    }


def train_and_evaluate_all():
    """Train all 10 regression models, cross-validate top 2, tune ensembles, and save artifacts."""
    X_train, X_test, y_train, y_test, feature_names = dl.split_regression()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = get_baseline_models()
    results = []
    fitted_models = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)

        r2 = float(r2_score(y_test, preds))
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        mae = float(mean_absolute_error(y_test, preds))

        results.append({
            "Algorithm": name,
            "R2": round(r2, 6),
            "RMSE": round(rmse, 6),
            "MAE": round(mae, 6),
        })
        fitted_models[name] = model

    df_results = pd.DataFrame(results).sort_values(by="R2", ascending=False).reset_index(drop=True)
    df_results.index += 1
    df_results.to_csv(C.REPORTS_DIR / "regression_comparison.csv", index_label="Rank")

    # 5-Fold Cross-Validation for the Top 2 Models
    top_2 = df_results["Algorithm"].head(2).tolist()
    cv_records = {}
    for name in top_2:
        m = fitted_models[name]
        scores = cross_val_score(m, X_train_scaled, y_train, cv=5, scoring="r2")
        cv_records[name] = {
            "mean_cv_r2": float(round(scores.mean(), 4)),
            "std_cv_r2": float(round(scores.std(), 4)),
            "folds": [float(round(s, 4)) for s in scores],
        }

    # Hyperparameter Tuning on 2 models (Random Forest & Gradient Boosting)
    rf_grid = {"max_depth": [3, 5, 10], "n_estimators": [50, 100, 200]}
    gs_rf = GridSearchCV(RandomForestRegressor(random_state=C.RANDOM_STATE), rf_grid, cv=5, scoring="r2")
    gs_rf.fit(X_train_scaled, y_train)
    rf_tuned_r2 = float(round(r2_score(y_test, gs_rf.predict(X_test_scaled)), 4))

    gb_grid = {"learning_rate": [0.05, 0.1, 0.2], "n_estimators": [50, 100, 150]}
    gs_gb = GridSearchCV(GradientBoostingRegressor(random_state=C.RANDOM_STATE), gb_grid, cv=5, scoring="r2")
    gs_gb.fit(X_train_scaled, y_train)
    gb_tuned_r2 = float(round(r2_score(y_test, gs_gb.predict(X_test_scaled)), 4))

    tuning_records = [
        {"Model": "Random Forest Regressor", "Best Params": str(gs_rf.best_params_), "Tuned R2": rf_tuned_r2},
        {"Model": "Gradient Boosting Regressor", "Best Params": str(gs_gb.best_params_), "Tuned R2": gb_tuned_r2},
    ]

    # Best model selection (Ridge Regression)
    best_name = df_results["Algorithm"].iloc[0]
    best_model = fitted_models[best_name]
    best_preds = best_model.predict(X_test_scaled)
    residuals = y_test.values - best_preds

    # Feature importances for Ridge / Tree models
    importances = {}
    if hasattr(best_model, "coef_"):
        coefs = np.abs(best_model.coef_ if best_model.coef_.ndim == 1 else best_model.coef_[0])
        importances = dict(zip(feature_names, coefs.tolist()))
    elif hasattr(best_model, "feature_importances_"):
        importances = dict(zip(feature_names, best_model.feature_importances_.tolist()))

    # Build standalone pipeline for live predictor
    full_pipeline = Pipeline([
        ("scaler", scaler),
        ("regressor", best_model)
    ])

    saved_payload = {
        "model_name": best_name,
        "pipeline": full_pipeline,
        "scaler": scaler,
        "feature_names": feature_names,
        "comparison": df_results.to_dict("records"),
        "top2_cv": cv_records,
        "tuning": tuning_records,
        "test_actual": y_test.values.tolist(),
        "test_pred": best_preds.tolist(),
        "residuals": residuals.tolist(),
        "importances": importances,
    }

    joblib.dump(saved_payload, C.MODELS_DIR / "best_regressor.joblib")

    # Update reports/results.json
    results_path = C.REPORTS_DIR / "results.json"
    data = {}
    if results_path.exists():
        try:
            data = json.loads(results_path.read_text())
        except Exception:
            data = {}
    data["regression"] = {
        "comparison": df_results.to_dict("records"),
        "top2_cv": cv_records,
        "tuning": tuning_records,
        "best_model": best_name,
        "best_r2": float(df_results["R2"].iloc[0]),
    }
    results_path.write_text(json.dumps(data, indent=2))

    return saved_payload
