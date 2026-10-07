# 🏠 House Loan Amount Prediction
## Course 602 – Data Analytics using Python | BCA Final Year Project
**Faculty:** Dr. Snehal K Joshi

---

## 📁 Project Structure
```
HouseLoan602/
├── app.py                  ← Streamlit Dashboard (Main App)
├── eda.py                  ← Tasks 2, 4, 5: EDA & Visualization
├── preprocessing.py        ← Tasks 3, 4: Cleaning & Encoding
├── regression.py           ← Tasks 6, 7, 8: Regression Models
├── classification.py       ← Tasks 9, 10: Classification Models
├── evaluation.py           ← Tasks 11, 12: Evaluation & CV
├── requirements.txt        ← Python dependencies
├── README.md               ← This file
├── data/
│   └── Surat_HouseLoan_Dataset.csv
├── models/                 ← Saved .pkl models (auto-created)
├── notebooks/
│   └── House_Loan_Prediction_602.ipynb  ← Complete Jupyter notebook
└── utils/
    └── data_generator.py   ← Surat dataset generator
```

---

## ⚙️ Setup & Run

### Step 1 – Install Python 3.9+
Download from: https://www.python.org/downloads/

### Step 2 – Create Virtual Environment (Recommended)
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### Step 3 – Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4 – Run Streamlit App
```bash
streamlit run app.py
```
Opens at: **http://localhost:8501**

### Step 5 – Run Jupyter Notebook
```bash
pip install jupyter
cd notebooks
jupyter notebook House_Loan_Prediction_602.ipynb
```

---

## 📊 Task Coverage (All 13 Tasks)

| Task | Description | Marks | File |
|------|-------------|-------|------|
| 1  | Data Understanding | 5  | app.py / notebook |
| 2  | EDA: Uni/Bi/Multivariate | 25 | eda.py |
| 3  | Missing Data & Outliers | 10 | preprocessing.py |
| 4  | Spread of Data | 5  | eda.py |
| 5  | Automating EDA | 5  | eda.py |
| 6  | Regression Analysis | 10 | regression.py |
| 7  | Supervised Learning Models | 10 | regression.py |
| 8  | Overfitting Analysis | 5  | regression.py |
| 9  | Classification Task | 5  | classification.py |
| 10 | Classification Evaluation | 5  | classification.py |
| 11 | Model Evaluation | 5  | evaluation.py |
| 12 | Performance Interpretation | 5  | evaluation.py |
| 13 | Dashboard & Visualization | 5  | app.py |
| **Total** | | **100** | |

---

## 🎯 Dataset
- **Name:** Surat City House Loan Prediction Dataset
- **Records:** 614 rows × 16 columns
- **Source:** Generated/Kaggle format
- **Key Columns:** Loan_ID, Gender, Married, Dependents, Education, Self_Employed,
  ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History,
  Property_Area, Loan_Status

---

## 🚀 Dashboard Features
- ✅ 7 Navigation tabs covering all tasks
- ✅ Upload your own CSV or use built-in Surat dataset
- ✅ Dark/Light theme toggle
- ✅ Sidebar filters (Area, Income, Education)
- ✅ All charts interactive (Plotly)
- ✅ Real-time prediction with EMI calculator
- ✅ Download report and filtered data
- ✅ Cross-validation & hyperparameter tuning
- ✅ ROC curves, confusion matrices, feature importance
