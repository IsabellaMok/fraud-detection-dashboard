# 💳 Credit Card Fraud Detection Dashboard

A machine learning-powered dashboard that detects fraudulent credit card transactions, built to explore how financial institutions can use data science for fraud monitoring.

**Results:** 92% precision, 74% recall on identifying fraud across 284,807 real transactions.

---

## My Process

### Step 1: Understanding the Problem
I started by researching how fraud detection works in banking and chose the ULB Credit Card Fraud dataset from Kaggle — a real dataset of 284,807 European card transactions, with only 492 (0.17%) being fraud.

### Step 2: Exploratory Data Analysis
Before building anything, I explored the data to understand it:
- Checked for missing values (none found)
- Analyzed class distribution → discovered the extreme imbalance (99.83% normal vs 0.17% fraud)
- Visualized transaction amounts by class

**Why this matters:** this imbalance is the whole reason fraud detection is a hard ML problem — a model that always predicts "not fraud" would still be 99.8% "accurate" but completely useless.

![Class Distribution](screenshots/class_distribution.png)

### Step 3: Model Training
I trained a Random Forest classifier with `class_weight="balanced"` to force the model to pay more attention to the rare fraud cases, rather than being biased toward the majority class.

**Result:**
- Precision: 0.92 (few false alarms)
- Recall: 0.74 (caught 3 out of 4 fraud cases)
- F1-score: 0.82

**Why I prioritized precision:** in a real banking context, too many false fraud alerts erode customer trust and increase support costs. I chose an approach that keeps false alarms low while still catching most fraud — this is a real business tradeoff, not just a technical one.

### Step 4: Building the Dashboard
I built a Streamlit dashboard so the model could be used interactively — upload transaction data, get instant fraud predictions with confidence scores.

<img width="1527" height="773" alt="image" src="https://github.com/user-attachments/assets/a4b202a1-5914-46a5-bcd4-2d34926ce113" />
<img width="1520" height="585" alt="image" src="https://github.com/user-attachments/assets/0a3c882c-03a4-4e11-b817-5b2bad94e1aa" />


---

## Key Learnings
- Class imbalance handling in ML (why accuracy alone is misleading)
- Precision vs recall as a business decision, not just a technical one
- End-to-end ML project structure: data → model → deployable interface
- Git/GitHub workflow, including troubleshooting a real issue (GitHub's 100MB file size limit rejected my dataset push — learned to properly use `.gitignore` and document reproduction steps instead)

## Dataset
[Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle (ULB Machine Learning Group)
- 284,807 transactions over 2 days (European cardholders, September 2013)
- Only 492 fraud cases (0.17% of the dataset) — highly imbalanced
- 30 features (Time, Amount, and 28 PCA-transformed anonymized features)

## How to Run
1. Clone this repo
