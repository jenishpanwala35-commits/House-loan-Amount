"""
preprocessing.py – Data Cleaning, Encoding, Outlier Handling
Course 602 – Data Analytics using Python | Task 3 & Task 4
"""
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_data(filepath: str) -> pd.DataFrame:
    """Load dataset from CSV file."""
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Dataset loaded: {df.shape[0]} rows × {df.shape[1]} cols")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def show_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Return missing value summary."""
    miss = df.isnull().sum()
    miss_pct = (miss / len(df) * 100).round(2)
    result = pd.DataFrame({'Missing_Count': miss, 'Missing_%': miss_pct})
    return result[result['Missing_Count'] > 0]


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Task 3 – Handle missing values:
    • Categorical → Mode
    • Numerical   → Median
    """
    df = df.copy()
    for col in df.select_dtypes(include='object').columns:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            logger.info(f"  {col}: filled with mode = '{mode_val}'")
    for col in df.select_dtypes(include=np.number).columns:
        if df[col].isnull().any():
            med_val = df[col].median()
            df[col].fillna(med_val, inplace=True)
            logger.info(f"  {col}: filled with median = {med_val:.2f}")
    return df


def cap_outliers_iqr(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """
    Task 3 – Detect & cap outliers using IQR method.
    Values beyond Q1-1.5*IQR or Q3+1.5*IQR are capped.
    """
    df = df.copy()
    for col in cols:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            n_out = ((df[col] < lower) | (df[col] > upper)).sum()
            df[col] = df[col].clip(lower, upper)
            logger.info(f"  {col}: {n_out} outliers capped [{lower:.1f}, {upper:.1f}]")
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode all categorical columns."""
    df = df.copy()
    mappings = {
        'Gender'       : {'Male': 1, 'Female': 0},
        'Married'      : {'Yes': 1, 'No': 0},
        'Education'    : {'Graduate': 1, 'Not Graduate': 0},
        'Self_Employed': {'Yes': 1, 'No': 0},
        'Property_Area': {'Urban': 2, 'Semiurban': 1, 'Rural': 0},
        'Loan_Status'  : {'Y': 1, 'N': 0},
        'Dependents'   : {'0': 0, '1': 1, '2': 2, '3+': 3},
    }
    for col, mapping in mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0).astype(int)
    return df


def get_features_targets(df: pd.DataFrame):
    """Return X, y_regression, y_classification, feature_names."""
    drop_cols = ['LoanAmount', 'Loan_Status', 'Loan_ID', 'Applicant_Name',
                 'Locality', 'Property_Type']
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X  = df[feature_cols].values
    yr = df['LoanAmount'].values
    yc = df['Loan_Status'].values
    return X, yr, yc, feature_cols


def full_pipeline(df: pd.DataFrame):
    """
    Run complete preprocessing pipeline.
    Returns: X, y_reg, y_cls, feature_names, clean_df
    """
    logger.info("=== Preprocessing Pipeline Start ===")
    df = impute_missing(df)
    df = cap_outliers_iqr(df, ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount'])
    df = encode_features(df)
    X, yr, yc, feat = get_features_targets(df)
    logger.info(f"Pipeline done → X{X.shape}, y_reg{yr.shape}, y_cls{yc.shape}")
    return X, yr, yc, feat, df
