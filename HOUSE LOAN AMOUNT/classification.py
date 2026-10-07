"""
classification.py – Classification Models (Tasks 9, 10)
Course 602 – Data Analytics using Python
"""
import numpy as np
import joblib, os, logging
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, roc_curve, auc,
                              classification_report)

logger = logging.getLogger(__name__)


def train_classification_models(X, y, feature_names, models_dir='models'):
    """
    Task 9 – Build classification models:
    • Logistic Regression
    • Random Forest Classifier
    Task 10 – Evaluate: Accuracy, Precision, Recall, F1, CM, ROC
    """
    os.makedirs(models_dir, exist_ok=True)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    results = {'metrics': {}, 'roc': {}}

    for name, model in [
        ('Logistic Regression', LogisticRegression(max_iter=1000, random_state=42)),
        ('Random Forest',       RandomForestClassifier(n_estimators=100, max_depth=10,
                                                       random_state=42, n_jobs=-1)),
    ]:
        model.fit(X_tr, y_tr)
        pred  = model.predict(X_te)
        proba = model.predict_proba(X_te)[:, 1]

        results['metrics'][name] = {
            'accuracy' : round(accuracy_score(y_te, pred), 4),
            'precision': round(precision_score(y_te, pred, zero_division=0), 4),
            'recall'   : round(recall_score(y_te, pred, zero_division=0), 4),
            'f1'       : round(f1_score(y_te, pred, zero_division=0), 4),
            'report'   : classification_report(y_te, pred, output_dict=True),
        }

        fpr, tpr, _ = roc_curve(y_te, proba)
        results['roc'][name] = {
            'fpr': fpr.tolist(), 'tpr': tpr.tolist(),
            'auc': round(auc(fpr, tpr), 4),
        }

        if name == 'Random Forest':
            results['rf_cm']      = confusion_matrix(y_te, pred)
            results['rf_pred']    = pred
            results['rf_actual']  = y_te
            results['rf_model']   = model
            results['feature_imp']= model.feature_importances_.tolist()

        logger.info(f"{name}: Acc={results['metrics'][name]['accuracy']}")

    joblib.dump(results['rf_model'], f'{models_dir}/rf_classifier.pkl')
    return results
