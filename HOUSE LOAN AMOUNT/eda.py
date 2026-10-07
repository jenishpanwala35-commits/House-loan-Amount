"""
eda.py – Exploratory Data Analysis (Tasks 2, 4, 5)
Course 602 – Data Analytics using Python
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


# ─────────────────────────────────────────────────────────────────────────────
# TASK 5 – Automated EDA Functions
# ─────────────────────────────────────────────────────────────────────────────

def automated_summary(df: pd.DataFrame) -> dict:
    """Task 5 – Reusable automated summary function."""
    return {
        'shape'    : df.shape,
        'dtypes'   : df.dtypes,
        'describe' : df.describe(include='all'),
        'nulls'    : df.isnull().sum(),
        'null_pct' : (df.isnull().sum() / len(df) * 100).round(2),
    }


def detect_outliers(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Task 5 – Detect outliers using IQR method."""
    rows = []
    for col in cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            Q1, Q3 = df[col].quantile([0.25, 0.75])
            IQR    = Q3 - Q1
            mask   = (df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)
            rows.append({'Column': col, 'Q1': round(Q1,2), 'Q3': round(Q3,2),
                         'IQR': round(IQR,2), 'Lower_Fence': round(Q1-1.5*IQR,2),
                         'Upper_Fence': round(Q3+1.5*IQR,2), 'Outlier_Count': int(mask.sum())})
    return pd.DataFrame(rows)


def correlation_report(df: pd.DataFrame) -> pd.DataFrame:
    """Task 5 – Correlation report for numerical columns."""
    return df.select_dtypes(include=np.number).corr().round(3)


# ─────────────────────────────────────────────────────────────────────────────
# TASK 4 – Spread of Data
# ─────────────────────────────────────────────────────────────────────────────

def spread_of_data(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """Task 4 – Mean, Median, Std, Skewness, Kurtosis."""
    rows = []
    for col in cols:
        if col in df.columns:
            s = df[col].dropna()
            rows.append({
                'Feature'  : col,
                'Mean'     : round(s.mean(), 2),
                'Median'   : round(s.median(), 2),
                'Std Dev'  : round(s.std(), 2),
                'Skewness' : round(s.skew(), 3),
                'Kurtosis' : round(s.kurtosis(), 3),
                'Distribution': 'Normal (|skew|<0.5)' if abs(s.skew()) < 0.5
                                 else ('Mildly Skewed' if abs(s.skew()) < 1
                                       else 'Highly Skewed'),
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# TASK 2 – Univariate Analysis
# ─────────────────────────────────────────────────────────────────────────────

def plot_univariate(df: pd.DataFrame, col: str) -> go.Figure:
    """Histogram + Boxplot side by side."""
    clean = df[col].dropna()
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=[f'{col} – Histogram', f'{col} – Boxplot'])
    fig.add_trace(go.Histogram(x=clean, nbinsx=40, marker_color='#2563EB',
                               name='Histogram'), row=1, col=1)
    fig.add_trace(go.Box(y=clean, boxpoints='all', marker_color='#7C3AED',
                         name='Boxplot', jitter=0.3, pointpos=-1.8), row=1, col=2)
    fig.update_layout(height=380, showlegend=False,
                      title_text=f'Task 2 – Univariate Analysis: {col}',
                      paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# TASK 2 – Bivariate Analysis
# ─────────────────────────────────────────────────────────────────────────────

def plot_income_vs_loan(df: pd.DataFrame) -> go.Figure:
    """Scatter: ApplicantIncome vs LoanAmount."""
    clean = df.dropna(subset=['ApplicantIncome', 'LoanAmount'])
    color_col = 'Property_Area' if 'Property_Area' in df.columns else None
    fig = px.scatter(clean, x='ApplicantIncome', y='LoanAmount',
                     color=color_col, trendline='ols',
                     title='Bivariate – ApplicantIncome vs LoanAmount',
                     labels={'ApplicantIncome': 'Applicant Income (₹)',
                             'LoanAmount': 'Loan Amount (₹000)'},
                     color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(height=400)
    return fig


def plot_credit_vs_status(df: pd.DataFrame) -> go.Figure:
    """Bar: Credit_History vs Loan_Status."""
    tmp = df.dropna(subset=['Credit_History', 'Loan_Status'])
    grp = tmp.groupby(['Credit_History', 'Loan_Status']).size().reset_index(name='Count')
    fig = px.bar(grp, x='Credit_History', y='Count', color='Loan_Status',
                 barmode='group',
                 title='Bivariate – Credit History vs Loan Status',
                 color_discrete_map={'Y': '#16A34A', 'N': '#DC2626'})
    fig.update_layout(height=380)
    return fig


def plot_area_vs_status(df: pd.DataFrame) -> go.Figure:
    """Bar: Property_Area vs Loan_Status."""
    tmp = df.dropna(subset=['Property_Area', 'Loan_Status'])
    grp = tmp.groupby(['Property_Area', 'Loan_Status']).size().reset_index(name='Count')
    fig = px.bar(grp, x='Property_Area', y='Count', color='Loan_Status',
                 barmode='group',
                 title='Bivariate – Property Area vs Loan Status',
                 color_discrete_map={'Y': '#16A34A', 'N': '#DC2626'})
    fig.update_layout(height=380)
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Interactive correlation heatmap."""
    corr = df.select_dtypes(include=np.number).corr().round(3)
    fig = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r',
                    title='Correlation Matrix Heatmap – All Numeric Features',
                    aspect='auto')
    fig.update_layout(height=500)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# TASK 2 – Multivariate Analysis
# ─────────────────────────────────────────────────────────────────────────────

def plot_pairplot(df: pd.DataFrame) -> go.Figure:
    """Pair plot: Income, LoanAmount, Credit_History."""
    cols  = [c for c in ['ApplicantIncome','CoapplicantIncome','LoanAmount','Credit_History']
             if c in df.columns]
    color = 'Property_Area' if 'Property_Area' in df.columns else None
    clean = df[cols + ([color] if color else [])].dropna()
    fig   = px.scatter_matrix(clean, dimensions=cols, color=color,
                               title='Multivariate – Pair Plot across Property Area',
                               height=650,
                               color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_traces(marker_size=3, diagonal_visible=True)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# TASK 4 – Distribution & Normalization Plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_distribution_transform(df: pd.DataFrame, col: str) -> go.Figure:
    """Original vs Log-transformed distribution."""
    clean = df[col].dropna()
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=[f'Original: {col}', f'Log-Transformed: {col}'])
    fig.add_trace(go.Histogram(x=clean, nbinsx=40, marker_color='#2563EB',
                               name='Original'), row=1, col=1)
    fig.add_trace(go.Histogram(x=np.log1p(clean), nbinsx=40, marker_color='#16A34A',
                               name='Log Transform'), row=1, col=2)
    fig.update_layout(height=380, showlegend=False,
                      title_text=f'Task 4 – Normalization: {col}')
    return fig
