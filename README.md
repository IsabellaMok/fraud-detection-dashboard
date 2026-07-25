# Credit Card Fraud Detection Dashboard

A machine learning-powered dashboard that detects fraudulent credit card transactions in real time, built to explore how financial institutions can use data science for fraud monitoring.

## Problem
Financial institutions process millions of transactions daily, with fraud representing a tiny but costly fraction of them. This project explores how to build a system that can flag suspicious transactions accurately, without overwhelming teams with false alarms.

## Dataset
[Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle (ULB Machine Learning Group)
- 284,807 transactions over 2 days (European cardholders, Sept 2013)
- Only 492 fraud cases (0.17% of the dataset) — highly imbalanced

## Approach
1. **Exploratory Data Analysis** — examined class distribution and transaction amount patterns to understand the imbalance problem
2. **Model Training** — trained a Random Forest classifier with `class_weight="balanced"` to handle the extreme imbalance
3. **Dashboard** — built an interactive Streamlit app for uploading transaction data and viewing flagged results in real time

## Results
- **Precision: 0.92** — 92% of flagged transactions were genuinely fraudulent (low false alarm rate)
- **Recall: 0.74** — caught 74% of actual fraud cases
- **F1-score: 0.82**

## Key Learnings
- Class imbalance is the central challenge in fraud detection — accuracy alone is a misleading metric (a model that predicts "not fraud" every time would still be 99.8% accurate but useless)
- Precision vs recall is a real business tradeoff: too many false alarms erode customer trust, but missing fraud is costly. This project prioritized precision while maintaining reasonable recall.
- This mirrors real-world transaction monitoring approaches used by financial institutions for fraud/AML compliance (e.g., relevant to frameworks like BNM RMiT in Malaysia)

## How to Run
1. Clone this repo
2. Download `creditcard.csv` from the [Kaggle dataset link](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in a `data/` folder
3. Install dependencies: `pip install -r requirements.txt`
4. Train the model: `python train_model.py`
5. Run the dashboard: `streamlit run dashboard.py`

## Tech Stack
Python, pandas, scikit-learn, Streamlit, matplotlib, seaborn
