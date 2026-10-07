"""
regression.py – Regression Models (Tasks 6, 7, 8)
Course 602 – Data Analytics using Python
"""
import numpy as np
import joblib, os, logging
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


def regression_metrics(y_true, y_pred, label='Model') -> dict:
    """Task 11 – Compute MAE, MSE, RMSE, R²."""
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)
    logger.info(f"{label} → MAE={mae:.2f} MSE={mse:.2f} RMSE={rmse:.2f} R²={r2:.4f}")
    return {'MAE': round(mae,3), 'MSE': round(mse,3),
            'RMSE': round(rmse,3), 'R²': round(r2,4)}


def covariance_correlation(X, y, feature_names) -> dict:
    """Task 6 – Compute covariance and correlation."""
    cov_dict  = {}
    corr_dict = {}
    for i, name in enumerate(feature_names):
        cov_dict[name]  = round(float(np.cov(X[:, i], y)[0, 1]), 3)
        corr_dict[name] = round(float(np.corrcoef(X[:, i], y)[0, 1]), 3)
    return {'covariance': cov_dict, 'correlation': corr_dict}


def train_regression_models(X, y, feature_names, models_dir='models'):
    """
    Task 7 – Build & evaluate:
    (i)  Simple Linear Regression
    (ii) Multiple Linear Regression
    (iii) Random Forest Regressor
    Also: Ridge & Lasso (Task 8 regularization)
    """
    os.makedirs(models_dir, exist_ok=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    results = {'metrics': {}, 'predictions': {}}

    # ── (i) Simple Linear Regression (1 feature: ApplicantIncome) ──────────
    slr = LinearRegression()
    slr.fit(X_tr[:, :1], y_tr)
    slr_pred = slr.predict(X_te[:, :1])
    results['metrics']['Simple LR']    = regression_metrics(y_te, slr_pred, 'Simple LR')
    results['predictions']['Simple LR']= {'actual': y_te.tolist(), 'pred': slr_pred.tolist()}

    # ── (ii) Multiple Linear Regression ────────────────────────────────────
    mlr = LinearRegression()
    mlr.fit(X_tr, y_tr)
    mlr_pred = mlr.predict(X_te)
    results['metrics']['Multiple LR']    = regression_metrics(y_te, mlr_pred, 'Multiple LR')
    results['predictions']['Multiple LR']= {'actual': y_te.tolist(), 'pred': mlr_pred.tolist()}
    results['mlr_coef'] = dict(zip(feature_names, mlr.coef_.round(4).tolist()))

    # ── Random Forest Regressor ─────────────────────────────────────────────
    rfr = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    rfr.fit(X_tr, y_tr)
    rfr_pred = rfr.predict(X_te)
    results['metrics']['Random Forest']    = regression_metrics(y_te, rfr_pred, 'Random Forest')
    results['predictions']['Random Forest']= {'actual': y_te.tolist(), 'pred': rfr_pred.tolist()}
    results['rf_importance']               = rfr.feature_importances_.tolist()

    # ── Ridge & Lasso (Task 8 – Regularization) ────────────────────────────
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_tr, y_tr)
    results['metrics']['Ridge (L2)'] = regression_metrics(y_te, ridge.predict(X_te), 'Ridge')

    lasso = Lasso(alpha=0.5, max_iter=10000)
    lasso.fit(X_tr, y_tr)
    results['metrics']['Lasso (L1)'] = regression_metrics(y_te, lasso.predict(X_te), 'Lasso')

    # ── Learning Curves (Task 8 – Overfitting analysis) ────────────────────
    sizes, tr_sc, val_sc = learning_curve(
        RandomForestRegressor(n_estimators=50, random_state=42),
        X, y, cv=5, scoring='r2',
        train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1)
    results['learning_curves'] = {
        'sizes': sizes.tolist(),
        'train': tr_sc.mean(axis=1).tolist(),
        'val'  : val_sc.mean(axis=1).tolist(),
    }

    # ── Train/Test Error Comparison ─────────────────────────────────────────
    results['train_test_errors'] = {
        'train_rmse': round(np.sqrt(mean_squared_error(y_tr, rfr.predict(X_tr))), 3),
        'test_rmse' : round(np.sqrt(mean_squared_error(y_te, rfr_pred)), 3),
    }

    # Save models
    joblib.dump(rfr, f'{models_dir}/rf_regressor.pkl')
    joblib.dump(mlr, f'{models_dir}/mlr_model.pkl')
    logger.info("Models saved.")

    results['X_test']  = X_te
    results['y_test']  = y_te
    results['rfr_model'] = rfr
    return results
