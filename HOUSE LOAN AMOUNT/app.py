
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings, logging, os, sys
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))

from preprocessing import load_data, show_missing, impute_missing, cap_outliers_iqr, encode_features, get_features_targets
from eda import (automated_summary, detect_outliers, correlation_report, spread_of_data,
                 plot_univariate, plot_income_vs_loan, plot_credit_vs_status,
                 plot_area_vs_status, plot_correlation_heatmap, plot_pairplot,
                 plot_distribution_transform)
from regression import regression_metrics, covariance_correlation, train_regression_models
from classification import train_classification_models
from evaluation import cross_validate_models, hyperparameter_tuning, business_interpretation
from utils.data_generator import generate_surat_loan_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Page Setup ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="House Loan Prediction | Course 602",
    page_icon="🏠", layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS Theming ──────────────────────────────────────────────────────────────
def inject_css():
    # Fixed Dark Theme
    bg   = "#0F172A"
    card = "#1E293B"
    txt  = "#F1F5F9"
    acc  = "#3B82F6"
    st.markdown(f"""<style>
    .main {{ background-color:{bg}; color:{txt}; }}
    .stApp {{ background-color:{bg}; }}
    .task-header {{ background:linear-gradient(90deg,#1E3A5F,{acc});
                    color:white; padding:14px 22px; border-radius:10px;
                    font-size:17px; font-weight:700; margin:12px 0 8px 0; }}
    .kpi-card {{ background:{card}; border-radius:12px; padding:18px 14px;
                 text-align:center; border:1px solid {acc}33; }}
    .kpi-val  {{ font-size:28px; font-weight:800; color:{acc}; }}
    .kpi-lbl  {{ font-size:13px; color:{txt}88; }}
    .insight  {{ background:{card}; border-left:4px solid {acc};
                 padding:12px 16px; border-radius:6px; margin:8px 0; }}
    .stTabs [data-baseweb="tab-list"] {{ gap:6px; }}
    .stTabs [data-baseweb="tab"] {{ border-radius:8px; padding:6px 16px; font-weight:600; }}
    </style>""", unsafe_allow_html=True)

# Apply dark theme immediately on load
inject_css()
def task_header(text):
    st.markdown(f'<div class="task-header">{text}</div>', unsafe_allow_html=True)

def insight(text):
    st.markdown(f'<div class="insight">💡 {text}</div>', unsafe_allow_html=True)

def kpi(label, value, col):
    col.markdown(f'<div class="kpi-card"><div class="kpi-val">{value}</div><div class="kpi-lbl">{label}</div></div>', unsafe_allow_html=True)


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏠 House Loan Prediction") 
    st.divider()
    st.subheader("📂 Dataset")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    use_sample = st.checkbox("Use Surat Sample Dataset", value=True)
    st.divider()

    st.caption("All charts are interactive – hover, zoom & pan")


# ─── Load Data ────────────────────────────────────────────────────────────────
@st.cache_data
def get_data(uploaded_file=None):
    if uploaded_file:
        return pd.read_csv(uploaded_file)
    return generate_surat_loan_data()

try:
    raw_df = get_data(uploaded if uploaded else None)
except Exception as e:
    st.error(f"Data error: {e}")
    st.stop()

# No filters – use full dataset
df_filtered = raw_df.copy()

# ─── Main Navigation ──────────────────────────────────────────────────────────
TABS = ["📊 Task 1–2: Data & EDA", "🧹 Task 3–5: Cleaning & Auto EDA",
        "📈 Task 6–8: Regression", "🎯 Task 9–10: Classification",
       "🏆 Task 11–12: Evaluation", "🔮 Task 13: Predict"]
tabs = st.tabs(TABS)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – DATA UNDERSTANDING + EDA
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    task_header("📊 Task 1: Data Understanding | Task 2: EDA")
    sub_tabs = st.tabs(["Task 1: Understanding", "Task 2A: Univariate",
                         "Task 2B: Bivariate", "Task 2C: Multivariate"])

    # ── Task 1 ──────────────────────────────────────────────────────────────
    with sub_tabs[0]:
        st.subheader("Task 1 – Data Understanding")
        c1,c2,c3,c4,c5 = st.columns(5)
        kpi("Total Records",  raw_df.shape[0], c1)
        kpi("Features",       raw_df.shape[1], c2)
        kpi("Missing Values", int(raw_df.isnull().sum().sum()), c3)
        kpi("Approved Loans", f"{(raw_df['Loan_Status']=='Y').mean()*100:.1f}%", c4)
        kpi("Avg Loan (₹K)",  f"{raw_df['LoanAmount'].median():.0f}", c5)

        st.markdown("**First 5 Rows:**")
        st.dataframe(raw_df.head(), use_container_width=True)
        st.markdown("**Last 5 Rows:**")
        st.dataframe(raw_df.tail(), use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Data Types & Non-Null Counts:**")
            dt_df = pd.DataFrame({"Column": raw_df.columns,
                                   "DType": raw_df.dtypes.values,
                                   "Non-Null": raw_df.notnull().sum().values})
            st.dataframe(dt_df, use_container_width=True)
        with col2:
            st.markdown("**Variable Classification:**")
            var_df = pd.DataFrame({
                "Variable": ["Loan_ID","Gender","Married","Dependents","Education",
                             "Self_Employed","ApplicantIncome","CoapplicantIncome",
                             "LoanAmount","Loan_Amount_Term","Credit_History",
                             "Property_Area","Loan_Status"],
                "Type": ["Qualitative-Nominal","Qualitative-Nominal","Qualitative-Nominal",
                         "Qualitative-Ordinal","Qualitative-Nominal","Qualitative-Nominal",
                         "Quantitative-Continuous","Quantitative-Continuous",
                         "Quantitative-Continuous","Quantitative-Discrete",
                         "Qualitative-Nominal","Qualitative-Nominal","Qualitative-Nominal"]
            })
            st.dataframe(var_df, use_container_width=True)

    # ── Task 2A – Univariate ─────────────────────────────────────────────────
    with sub_tabs[1]:
        st.subheader("Task 2A – Univariate Analysis")
        insight("Univariate analysis studies each variable independently using histograms and boxplots to understand shape, center, and spread.")
        for col in ["ApplicantIncome","LoanAmount","Loan_Amount_Term"]:
            fig = plot_univariate(df_filtered, col)
            st.plotly_chart(fig, use_container_width=True)

    # ── Task 2B – Bivariate ───────────────────────────────────────────────────
    with sub_tabs[2]:
        st.subheader("Task 2B – Bivariate Analysis")
        insight("Bivariate analysis examines relationships between two variables using scatter plots, bar charts, and correlation.")
        st.plotly_chart(plot_income_vs_loan(df_filtered), use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(plot_credit_vs_status(df_filtered), use_container_width=True)
        with col2:
            st.plotly_chart(plot_area_vs_status(df_filtered), use_container_width=True)
        st.plotly_chart(plot_correlation_heatmap(df_filtered), use_container_width=True)

    # ── Task 2C – Multivariate ────────────────────────────────────────────────
    with sub_tabs[3]:
        st.subheader("Task 2C – Multivariate Analysis")
        insight("Multivariate analysis studies 3+ variables simultaneously using pair plots and heatmaps to find complex patterns.")
        st.plotly_chart(plot_pairplot(df_filtered), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – CLEANING + AUTO EDA
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    task_header("🧹 Task 3: Missing Data & Outliers | Task 4: Spread | Task 5: Auto EDA")
    sub = st.tabs(["Task 3: Missing & Outliers","Task 4: Spread","Task 5: Auto EDA"])

    with sub[0]:
        st.subheader("Task 3 – Missing Data Detection & Handling")
        miss_df = show_missing(raw_df)
        if len(miss_df) > 0:
            st.dataframe(miss_df, use_container_width=True)
            fig = px.bar(miss_df.reset_index(), x="index", y="Missing_%",
                         color="Missing_%", color_continuous_scale="Reds",
                         title="Missing Values % per Column")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("No missing values found!")

        insight("Strategy: Categorical → Mode imputation | Numerical → Median imputation")
        st.markdown("**After Imputation:**")
        df_clean = impute_missing(raw_df.copy())
        st.success(f"Missing values after cleaning: {df_clean.isnull().sum().sum()}")

        st.subheader("Outlier Detection (IQR Boxplots)")
        num_cols = ["ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term"]
        fig2 = make_subplots(rows=2, cols=2, subplot_titles=num_cols)
        for i, col in enumerate(num_cols):
            r, c = divmod(i, 2)
            fig2.add_trace(go.Box(y=df_filtered[col].dropna(), name=col,
                                   boxpoints="outliers", marker_color="#2563EB"), row=r+1, col=c+1)
        fig2.update_layout(height=520, title_text="Outlier Detection – IQR Boxplots", showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

        out_df = detect_outliers(df_filtered, num_cols)
        st.dataframe(out_df, use_container_width=True)
        insight("Impact of outliers: Extreme income values inflate regression coefficients, increase RMSE, and produce incorrect loan amount predictions. Capping using IQR ensures stable model training.")

    with sub[1]:
        st.subheader("Task 4 – Spread of Data")
        feat_cols = ["ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Amount_Term"]
        spread_df = spread_of_data(df_filtered, feat_cols)
        st.dataframe(spread_df, use_container_width=True)

        sel = st.selectbox("Select feature for distribution analysis:", feat_cols)
        fig = plot_distribution_transform(df_filtered, sel)
        st.plotly_chart(fig, use_container_width=True)
        skew_val = df_filtered[sel].skew()
        insight(f"{sel} – Skewness = {skew_val:.3f}. {'Right-skewed (positive): log transformation normalizes distribution.' if skew_val > 0.5 else ('Left-skewed (negative): reflect then log.' if skew_val < -0.5 else 'Approximately normal distribution.')}")

    with sub[2]:
        st.subheader("Task 5 – Automated EDA Functions")
        st.markdown("**`describe()` Summary:**")
        st.dataframe(df_filtered.describe(include="all").round(2), use_container_width=True)
        st.markdown("**`correlation_report()` Output:**")
        st.dataframe(correlation_report(df_filtered), use_container_width=True)
        st.markdown("**`detect_outliers()` Output:**")
        st.dataframe(detect_outliers(df_filtered, feat_cols), use_container_width=True)
        insight("These reusable functions (automated_summary, detect_outliers, correlation_report) follow Task 5 requirements using describe(), info(), isnull(), and corr().")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – REGRESSION
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    task_header("📈 Task 6: Regression Analysis | Task 7: Models | Task 8: Overfitting")
    try:
        @st.cache_data
        def run_regression(df):
            X, yr, yc, feat, df_c = _full_pipeline(df)
            res = train_regression_models(X, yr, feat, "models")
            cov_cor = covariance_correlation(X, yr, feat)
            return X, yr, feat, res, cov_cor, df_c

        def _full_pipeline(df):
            from preprocessing import full_pipeline
            return full_pipeline(df.copy())

        X, yr, feat, reg_res, cov_cor, df_clean2 = run_regression(df_filtered)

        sub_r = st.tabs(["Task 6: Cov/Corr","Task 7: Models","Task 8: Overfit"])

        with sub_r[0]:
            st.subheader("Task 6 – Covariance & Correlation Analysis")
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Covariance with LoanAmount:**")
                cov_df = pd.DataFrame(list(cov_cor["covariance"].items()), columns=["Feature","Covariance"])
                st.dataframe(cov_df.sort_values("Covariance",ascending=False), use_container_width=True)
            with col2:
                st.markdown("**Correlation with LoanAmount:**")
                corr_df = pd.DataFrame(list(cov_cor["correlation"].items()), columns=["Feature","Correlation"])
                st.dataframe(corr_df.sort_values("Correlation",ascending=False), use_container_width=True)
            fig = px.bar(corr_df.sort_values("Correlation"), x="Correlation", y="Feature",
                         orientation="h", title="Feature Correlation with LoanAmount",
                         color="Correlation", color_continuous_scale="RdBu")
            st.plotly_chart(fig, use_container_width=True)

        with sub_r[1]:
            st.subheader("Task 7 – Model Performance Comparison")
            met_df = pd.DataFrame(reg_res["metrics"]).T.reset_index()
            met_df.columns = ["Model","MAE","MSE","RMSE","R²"]
            st.dataframe(met_df.round(3), use_container_width=True)

            fig = px.bar(met_df.melt(id_vars="Model"), x="Model", y="value",
                         color="variable", barmode="group",
                         title="Task 7 – Regression Metrics Comparison")
            st.plotly_chart(fig, use_container_width=True)

            preds = reg_res["predictions"]["Random Forest"]
            fig2 = px.scatter(x=preds["actual"], y=preds["pred"],
                              labels={"x":"Actual LoanAmount","y":"Predicted LoanAmount"},
                              title="Random Forest – Actual vs Predicted", trendline="ols",
                              color_discrete_sequence=["#2563EB"])
            fig2.add_shape(type="line",
                           x0=min(preds["actual"]), y0=min(preds["actual"]),
                           x1=max(preds["actual"]), y1=max(preds["actual"]),
                           line=dict(color="red", dash="dash"))
            st.plotly_chart(fig2, use_container_width=True)

            st.markdown("**Multiple LR Coefficients:**")
            coef_df = pd.DataFrame(list(reg_res["mlr_coef"].items()), columns=["Feature","Coefficient"])
            st.dataframe(coef_df.sort_values("Coefficient",ascending=False), use_container_width=True)

        with sub_r[2]:
            st.subheader("Task 8 – Overfitting & Underfitting Analysis")
            te = reg_res["train_test_errors"]
            c1, c2 = st.columns(2)
            c1.metric("Train RMSE", f"{te['train_rmse']:.2f}")
            c2.metric("Test RMSE",  f"{te['test_rmse']:.2f}")
            if te["test_rmse"] > te["train_rmse"] * 1.5:
                st.warning("⚠️ Possible OVERFITTING – significant gap between train and test error")
            else:
                st.success("✅ Good Fit – train and test errors are close")

            lc = reg_res["learning_curves"]
            fig_lc = go.Figure()
            fig_lc.add_trace(go.Scatter(x=lc["sizes"], y=lc["train"],
                                         mode="lines+markers", name="Training Score",
                                         line=dict(color="#2563EB", width=2)))
            fig_lc.add_trace(go.Scatter(x=lc["sizes"], y=lc["val"],
                                         mode="lines+markers", name="Validation Score",
                                         line=dict(color="#DC2626", width=2)))
            fig_lc.update_layout(title="Learning Curves – Overfitting / Underfitting",
                                  xaxis_title="Training Samples", yaxis_title="R² Score", height=420)
            st.plotly_chart(fig_lc, use_container_width=True)

            st.markdown("**Ridge vs Lasso Regularization:**")
            rl_df = pd.DataFrame({k: v for k,v in reg_res["metrics"].items()
                                   if "Ridge" in k or "Lasso" in k}).T.reset_index()
            rl_df.columns = ["Model","MAE","MSE","RMSE","R²"]
            st.dataframe(rl_df.round(3), use_container_width=True)
            insight("Ridge (L2) shrinks all coefficients. Lasso (L1) can zero-out features for built-in feature selection. Both reduce overfitting.")

            fi_df = pd.DataFrame({"Feature": feat, "Importance": reg_res["rf_importance"]}).sort_values("Importance")
            fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                            title="Feature Importance – Random Forest Regressor",
                            color="Importance", color_continuous_scale="Blues")
            st.plotly_chart(fig_fi, use_container_width=True)

    except Exception as e:
        st.error(f"Regression error: {e}")
        import traceback; st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 – CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    task_header("🎯 Task 9: Classification | Task 10: Evaluation")
    try:
        @st.cache_data
        def run_classification(df):
            from preprocessing import full_pipeline
            X, yr, yc, feat, _ = full_pipeline(df.copy())
            return train_classification_models(X, yc, feat, "models")

        cls_res = run_classification(df_filtered)

        st.subheader("Task 9 – Classification Models")
        insight("Loan_Status converted: Y (Approved) = 1 | N (Not Approved) = 0")

        col1, col2 = st.columns(2)
        for i, (mname, metrics) in enumerate(cls_res["metrics"].items()):
            with col1 if i == 0 else col2:
                st.markdown(f"**{mname}**")
                m1,m2,m3,m4 = st.columns(4)
                m1.metric("Accuracy",  f"{metrics['accuracy']:.3f}")
                m2.metric("Precision", f"{metrics['precision']:.3f}")
                m3.metric("Recall",    f"{metrics['recall']:.3f}")
                m4.metric("F1 Score",  f"{metrics['f1']:.3f}")

        st.subheader("Task 10 – Confusion Matrix")
        cm = cls_res["rf_cm"]
        fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                           labels=dict(x="Predicted",y="Actual"),
                           x=["Not Approved","Approved"], y=["Not Approved","Approved"],
                           title="Confusion Matrix – Random Forest Classifier")
        fig_cm.update_layout(height=420)
        st.plotly_chart(fig_cm, use_container_width=True)

        st.subheader("ROC Curves (Interactive)")
        fig_roc = go.Figure()
        for mname, roc in cls_res["roc"].items():
            fig_roc.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines",
                                          name=f"{mname} (AUC={roc['auc']:.3f})"))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random",
                                      line=dict(dash="dash", color="gray")))
        fig_roc.update_layout(title="ROC Curves – Logistic Regression vs Random Forest",
                               xaxis_title="False Positive Rate",
                               yaxis_title="True Positive Rate", height=430)
        st.plotly_chart(fig_roc, use_container_width=True)

    except Exception as e:
        st.error(f"Classification error: {e}")
        import traceback; st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 – EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    task_header("🏆 Task 11: Regression Evaluation | Task 12: Interpretation")
    try:
        @st.cache_data
        def run_evaluation(df):
            from preprocessing import full_pipeline
            X, yr, yc, feat, _ = full_pipeline(df.copy())
            cv   = cross_validate_models(X, yr)
            hp   = hyperparameter_tuning(X, yr)
            return cv, hp, X, yr, feat

        cv_res, hp_res, X_ev, yr_ev, feat_ev = run_evaluation(df_filtered)

        st.subheader("Task 11 – Cross-Validation (K=5 Fold)")
        cv_df = pd.DataFrame(cv_res).T.reset_index()
        cv_df.columns = ["Model","CV Mean R²","CV Std","Scores"]
        st.dataframe(cv_df[["Model","CV Mean R²","CV Std"]], use_container_width=True)
        fig = px.bar(cv_df, x="Model", y="CV Mean R²", error_y="CV Std",
                     title="5-Fold Cross-Validation R² Scores",
                     color="CV Mean R²", color_continuous_scale="Greens")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Task 12 – Hyperparameter Tuning (GridSearchCV)")
        col1, col2 = st.columns(2)
        col1.json(hp_res["Best Parameters"])
        col2.metric("Best CV R²", hp_res["Best CV R²"])

        st.subheader("Task 12 – Business Interpretation")
        reg_res_ev = train_regression_models(X_ev, yr_ev, feat_ev, "models")
        interp = business_interpretation(reg_res_ev["metrics"])
        for row in interp:
            with st.expander(f"{row['Model']} – RMSE: {row['RMSE (₹000)']} | R²: {row['R² Score']}"):
                st.write(row["Meaning"])

    except Exception as e:
        st.error(f"Evaluation error: {e}")
        import traceback; st.code(traceback.format_exc())


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 – PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    task_header("🔮 Task 13: Live Loan Amount Prediction")
    st.subheader("Enter Applicant Details to Predict Loan Amount & Approval")

    col1, col2, col3 = st.columns(3)
    with col1:
        gender       = st.selectbox("Gender",     ["Male","Female"])
        married      = st.selectbox("Married",    ["Yes","No"])
        dependents   = st.selectbox("Dependents", ["0","1","2","3+"])
        education    = st.selectbox("Education",  ["Graduate","Not Graduate"])
        self_emp     = st.selectbox("Self Employed", ["No","Yes"])
    with col2:
        app_income   = st.number_input("Applicant Income (₹/month)", 0, 200000, 35000, 1000)
        co_income    = st.number_input("Coapplicant Income (₹/month)", 0, 100000, 0, 1000)
        loan_term    = st.selectbox("Loan Term (months)", [360,240,180,120,84,60])
        credit       = st.selectbox("Credit History", ["1 – Good","0 – Bad"])
    with col3:
        prop_area    = st.selectbox("Property Area", ["Urban","Semiurban","Rural"])
        interest_rate = st.slider("Interest Rate (% p.a.)", 6.0, 15.0, 8.5, 0.1)

    if st.button("🔮 Predict Loan Amount & Approval", type="primary", use_container_width=True):
        try:
            from preprocessing import full_pipeline
            from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

            X_t, yr_t, yc_t, feat_t, _ = full_pipeline(df_filtered.copy())

            rfr_p = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            rfr_p.fit(X_t, yr_t)
            rfc_p = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            rfc_p.fit(X_t, yc_t)

            credit_v  = 1 if "1" in credit else 0
            dep_v     = 3 if "+" in dependents else int(dependents)
            area_v    = 2 if prop_area=="Urban" else (1 if prop_area=="Semiurban" else 0)
            edu_v     = 1 if education=="Graduate" else 0
            mar_v     = 1 if married=="Yes" else 0
            gen_v     = 1 if gender=="Male" else 0
            se_v      = 1 if self_emp=="Yes" else 0

            input_data = np.array([[gen_v, mar_v, dep_v, edu_v, se_v,
                                     app_income, co_income, float(loan_term),
                                     credit_v, area_v]])

            if input_data.shape[1] == len(feat_t):
                pred_amount   = rfr_p.predict(input_data)[0]
                pred_approval = rfc_p.predict(input_data)[0]
                pred_proba    = rfc_p.predict_proba(input_data)[0][1]
            else:
                pad = np.zeros((1, len(feat_t)))
                pad[0, :min(10, len(feat_t))] = input_data[0, :min(10, len(feat_t))]
                pred_amount   = rfr_p.predict(pad)[0]
                pred_approval = rfc_p.predict(pad)[0]
                pred_proba    = rfc_p.predict_proba(pad)[0][1]

            monthly_rate = (interest_rate/100) / 12
            emi = (pred_amount*1000 * monthly_rate * (1+monthly_rate)**loan_term) / \
                  ((1+monthly_rate)**loan_term - 1) if monthly_rate > 0 else pred_amount*1000/loan_term

            st.divider()
            r1,r2,r3,r4 = st.columns(4)
            r1.metric("💰 Predicted Loan", f"₹{pred_amount*1000:,.0f}")
            r2.metric("📅 Monthly EMI",    f"₹{emi:,.0f}")
            r3.metric("✅ Approval",       "APPROVED" if pred_approval==1 else "REJECTED")
            r4.metric("📊 Approval Prob",  f"{pred_proba*100:.1f}%")

            if pred_approval == 1:
                st.success(f"🎉 Loan APPROVED! Predicted amount: ₹{pred_amount*1000:,.0f} with {pred_proba*100:.1f}% confidence")
            else:
                st.error(f"❌ Loan likely NOT APPROVED. Approval probability: {pred_proba*100:.1f}%")
                st.info("💡 Tip: Improving credit history or adding a co-applicant can significantly increase approval chances.")

        except Exception as e:
            st.error(f"Prediction error: {e}")
            import traceback; st.code(traceback.format_exc())


