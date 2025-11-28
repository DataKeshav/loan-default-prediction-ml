# Loan Default Prediction – Summary & Insights

## 1. Problem Statement

The goal of this project is to **predict whether a borrower will default on a loan** using historical data of 255,347 loans and 16 features (Age, Income, CreditScore, DTIRatio, EmploymentType, etc.).  
This helps a lender **identify high‑risk applicants before approving loans**.

---

## 2. Data Overview

- Total records: **255,347**
- Features: **16** (9 numeric, 7 categorical)
- Target: **Default** (0 = No Default, 1 = Default)
- Default rate: **11.61%** (imbalanced dataset)

Key columns:
- Financial: `Income`, `LoanAmount`, `CreditScore`, `InterestRate`, `LoanTerm`, `DTIRatio`, `NumCreditLines`
- Demographic/Behavioral: `Age`, `Education`, `EmploymentType`, `MaritalStatus`
- Flags: `HasMortgage`, `HasDependents`, `HasCoSigner`

---

## 3. Modeling Approach

### 3.1 Preprocessing

- Dropped ID/date columns (if any)
- Split into:
  - **X**: all features except `Default`
  - **y**: `Default` (0/1)
- Encoded categorical variables using **LabelEncoder**
- Train‑test split: **80% train / 20% test** with `stratify=y` to preserve default rate

### 3.2 Models Tried

1. **Logistic Regression** (baseline)
2. **Random Forest**
3. **Gradient Boosting**

Initially, all models were trained with **default class weights**, then I explored  
`class_weight='balanced'` and **threshold tuning** with Logistic Regression to handle imbalance.

---

## 4. Key Results

### 4.1 Initial Gradient Boosting (unbalanced)

- Accuracy ≈ **88.6%**
- For **Default (1)**:
  - Recall ≈ **5%**
  - F1-score ≈ **9%**

> Interpretation:  
> The model predicted “No Default” for ~99% of cases, giving high accuracy but **almost never catching defaulters**.  
> This is a classic problem with **imbalanced data**.

---

### 4.2 Logistic Regression with `class_weight='balanced'` (Threshold = 0.5)

- Accuracy ≈ **67.6%**
- For **Default (1)**:
  - Precision ≈ **22%**
  - Recall ≈ **69%**
  - F1-score ≈ **33%**

> Interpretation:  
> After using class weighting, the model became much more sensitive to defaulters:
> - It **caught ~69% of actual defaults** (big improvement)
> - It produced more false positives, so overall accuracy dropped  
> This is often a better trade‑off for credit risk, where **missing a defaulter is very costly**.

---

### 4.3 Threshold Tuning (Balanced Logistic Regression)

Using predicted probabilities from the balanced Logistic Regression:

- At **threshold 0.5**:
  - Default recall ≈ **69%**, precision ≈ **22%**

- At **threshold 0.4**:
  - Default recall ≈ **82%**, precision ≈ **18%**

- At **threshold 0.3**:
  - Default recall ≈ **91%**, precision ≈ **15%**

> Lowering the threshold **increases recall** (catch more defaulters)  
> but **reduces precision and accuracy** (more good customers flagged as risky).  
> This demonstrates the **precision–recall trade‑off** clearly.

---

## 5. Business Interpretation

- With the **original high‑accuracy model**, the bank would **miss ~95% of defaulters**, which is unacceptable.
- With **balanced Logistic Regression + threshold tuning**, the bank can:
  - Catch **70–90% of potential defaulters**
  - Accept more false alarms in exchange for **better risk control**

A lender could choose:
- Threshold **0.5** for a **balanced** strategy  
- Threshold **0.4** or **0.3** for a **more conservative risk policy** (prioritizing not missing any risky customers)

---

## 6. Key Learnings

1. **Accuracy is not enough** for imbalanced problems like loan default.  
   Metrics like **recall and F1-score for the positive class (Default)** are more important.

2. **Class imbalance** causes models to favor the majority class.  
   Techniques like `class_weight='balanced'` can significantly improve recall for the minority class.

3. **Threshold tuning** on predicted probabilities is a powerful way to control the  
   trade‑off between catching more defaulters (recall) and reducing false alarms (precision).

4. Simple models like **Logistic Regression**, when properly tuned, can perform  
   competitively and are easier to interpret than complex models.

---

## 7. Possible Next Steps

- Try **Random Forest with `class_weight='balanced'`** and compare with Logistic Regression.
- Use **SMOTE or other resampling techniques** to handle imbalance.
- Add **feature importance** and SHAP analysis to explain which factors drive default.
- Deploy the best model as an API or integrate it into a loan approval dashboard.

---
