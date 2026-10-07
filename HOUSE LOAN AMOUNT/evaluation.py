"""
evaluation.py – Model Evaluation: Cross-Validation & Interpretation (Tasks 11, 12)
Course 602 – Data Analytics using Python
"""
import numpy as np
import logging
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV

logger = logging.getLogger(__name__)


def cross_validate_models(X, y_reg):
    """Task 12 – 5-Fold Cross Validation."""
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest'    : RandomForestRegressor(n_estimators=50, random_state=42),
    }
    cv_results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X, y_reg, cv=5, scoring='r2', n_jobs=-1)
        cv_results[name] = {
            'CV Mean R²': round(scores.mean(), 4),
            'CV Std'    : round(scores.std(), 4),
            'Scores'    : scores.round(4).tolist(),
        }
        logger.info(f"{name} CV R² = {scores.mean():.4f} ± {scores.std():.4f}")
    return cv_results


def hyperparameter_tuning(X, y_reg):
    """Task 12 – GridSearchCV for best hyperparameters."""
    param_grid = {'n_estimators': [50, 100], 'max_depth': [5, 10, None]}
    gs = GridSearchCV(RandomForestRegressor(random_state=42),
                      param_grid, cv=3, scoring='r2', n_jobs=-1)
    gs.fit(X, y_reg)
    return {
        'Best Parameters': gs.best_params_,
        'Best CV R²'     : round(gs.best_score_, 4),
    }


def business_interpretation(metrics: dict) -> list:
    """Task 12 – Business meaning of model metrics in loan context."""
    insights = []
    for model, m in metrics.items():
        rmse = m.get('RMSE', 0)
        r2   = m.get('R²', 0)
        insights.append({
            'Model'        : model,
            'RMSE (₹000)'  : rmse,
            'R² Score'     : r2,
            'Meaning'      : (
                f"The {model} predicts loan amounts with an average error of ₹{rmse*1000:,.0f}. "
                f"It explains {r2*100:.1f}% of the variance in loan amounts. "
                + ("✅ Excellent fit." if r2 > 0.80
                   else ("✔ Good fit." if r2 > 0.60
                         else "⚠ Needs improvement."))
            )
        })
    return insights
