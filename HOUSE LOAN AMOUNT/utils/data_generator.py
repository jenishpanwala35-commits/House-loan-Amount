"""
utils/data_generator.py
Generates realistic Surat City House Loan Dataset
Course 602 – Data Analytics using Python
"""
import pandas as pd
import numpy as np

def generate_surat_loan_data(n=614, seed=42):
    np.random.seed(seed)

    urban    = ["Adajan","Vesu","Pal","Althan","Citylight","Athwa","Ghod Dod Road"]
    semi     = ["Katargam","Varachha","Udhna","Bhatar","Amroli","Limbayat"]
    rural    = ["Olpad","Kamrej","Sachin","Kim","Hazira","Bajipura"]

    male_n   = ["Rajesh Patel","Suresh Shah","Amit Desai","Vijay Modi","Nilesh Mehta",
                "Ketan Patel","Dhruv Shah","Harsh Desai","Ravi Joshi","Manish Trivedi",
                "Dipak Parmar","Sanjay Patel","Hitesh Shah","Vishal Mehta","Nirav Desai",
                "Chirag Patel","Bhavesh Shah","Jignesh Modi","Alpesh Patel","Tushar Shah"]
    female_n = ["Priya Patel","Neha Shah","Asha Desai","Kavita Modi","Hetal Mehta",
                "Mital Patel","Ritu Shah","Pooja Joshi","Nisha Trivedi","Sonal Parmar",
                "Foram Patel","Rupal Shah","Disha Mehta","Komal Desai","Jalpa Patel"]

    gender      = np.random.choice(["Male","Female"], n, p=[0.78,0.22])
    names       = [np.random.choice(male_n) if g=="Male" else np.random.choice(female_n) for g in gender]
    married     = np.random.choice(["Yes","No"], n, p=[0.67,0.33])
    dependents  = np.random.choice(["0","1","2","3+"], n, p=[0.55,0.18,0.17,0.10])
    education   = np.random.choice(["Graduate","Not Graduate"], n, p=[0.76,0.24])
    self_emp    = np.random.choice(["Yes","No"], n, p=[0.18,0.82])

    app_income  = np.array([np.random.randint(12000,85000) if e=="Graduate"
                             else np.random.randint(8000,40000) for e in education])
    co_income   = np.where(married=="Yes", np.random.randint(0,35000,n), 0)

    prop_area   = np.random.choice(["Urban","Semiurban","Rural"], n, p=[0.42,0.38,0.20])
    localities  = [np.random.choice(urban) if a=="Urban"
                   else (np.random.choice(semi) if a=="Semiurban"
                         else np.random.choice(rural)) for a in prop_area]

    amp = {"Urban":1.4,"Semiurban":1.1,"Rural":0.8}
    loan_amount = np.array([max(50, min(700,
                    int((app_income[i]+co_income[i])*np.random.uniform(2.5,6.0)*amp[prop_area[i]]/1000)))
                    for i in range(n)])

    term    = np.random.choice([360,240,180,120,84,60], n, p=[0.65,0.10,0.08,0.07,0.05,0.05])
    credit  = np.random.choice([1.0,0.0,np.nan], n, p=[0.78,0.17,0.05])
    prop_type = np.random.choice(["Apartment","Row House","Bungalow","Shop+Flat","Tenament"],
                                  n, p=[0.50,0.20,0.15,0.10,0.05])

    prob = np.clip(0.25 + 0.30*(credit==1) + 0.10*(app_income>30000)
                   + 0.08*(education=="Graduate") + 0.07*(married=="Yes")
                   + 0.08*(prop_area=="Semiurban") + 0.05*(prop_area=="Urban"), 0.05, 0.95)
    loan_status = np.where(np.random.rand(n) < prob, "Y", "N")

    # Inject realistic missing values
    idx = np.random.choice(n, 50, replace=False)
    gender_c  = gender.astype(object);  gender_c[idx[:8]]   = np.nan
    self_emp_c= self_emp.astype(object); self_emp_c[idx[8:18]]= np.nan
    loan_a_c  = loan_amount.astype(object); loan_a_c[idx[18:30]]= np.nan
    term_c    = term.astype(object);    term_c[idx[30:38]]  = np.nan
    credit_c  = credit.astype(object);  credit_c[idx[38:50]]= np.nan

    df = pd.DataFrame({
        "Loan_ID"          : [f"SURAT{str(i).zfill(5)}" for i in range(1,n+1)],
        "Applicant_Name"   : names,
        "Gender"           : gender_c,
        "Married"          : married,
        "Dependents"       : dependents,
        "Education"        : education,
        "Self_Employed"    : self_emp_c,
        "ApplicantIncome"  : app_income,
        "CoapplicantIncome": co_income,
        "LoanAmount"       : loan_a_c,
        "Loan_Amount_Term" : term_c,
        "Credit_History"   : credit_c,
        "Property_Area"    : prop_area,
        "Locality"         : localities,
        "Property_Type"    : prop_type,
        "Loan_Status"      : loan_status,
    })
    return df

if __name__ == "__main__":
    df = generate_surat_loan_data()
    df.to_csv("data/Surat_HouseLoan_Dataset.csv", index=False)
    print(f"Dataset saved: {df.shape}")
    print(df.head())
