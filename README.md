# 🏦 Bank Term Deposit Prediction System

A production-ready Machine Learning system that predicts whether a bank client will subscribe to a term deposit, built with an end-to-end ML pipeline and deployed via Streamlit.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.46-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6-F7931E?logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-3.1-189FDD)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📋 Overview

This project addresses a real-world banking challenge: **predicting which clients are most likely to subscribe to a term deposit** during direct marketing campaigns (phone calls). By identifying high-potential customers, the bank can optimize marketing spend and improve conversion rates.

### Key Features

- ✅ **End-to-End ML Pipeline** — From raw data to production predictions
- ✅ **XGBoost Model** — Tuned with RandomizedSearchCV, SMOTE for class imbalance
- ✅ **Custom Threshold Optimization** — Maximizes F1-Score for imbalanced data
- ✅ **Manual Prediction** — Interactive form for single client prediction
- ✅ **Bulk Prediction** — Upload CSV/Excel/JSON files for batch predictions
- ✅ **Sample Templates** — Download correctly-structured template files
- ✅ **Column Validation** — User-friendly error messages for malformed uploads
- ✅ **Dark Theme UI** — Premium, professional Streamlit interface

---

## 📊 Dataset

The [Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing) from UCI Machine Learning Repository.

| Property | Value |
|----------|-------|
| Rows | 45,211 |
| Features | 16 |
| Target | `y` (yes/no) |
| Class Balance | 88.3% No / 11.7% Yes |
| Source | Portuguese banking institution |

### Feature Descriptions

| Feature | Type | Description |
|---------|------|-------------|
| `age` | Numeric | Client's age |
| `job` | Categorical | Type of job |
| `marital` | Categorical | Marital status |
| `education` | Categorical | Education level |
| `default` | Binary | Has credit in default? |
| `balance` | Numeric | Yearly average balance (EUR) |
| `housing` | Binary | Has housing loan? |
| `loan` | Binary | Has personal loan? |
| `contact` | Categorical | Contact communication type |
| `day` | Numeric | Last contact day of the month |
| `month` | Categorical | Last contact month |
| `duration` | Numeric | Last contact duration (seconds) |
| `campaign` | Numeric | Contacts during this campaign |
| `pdays` | Numeric | Days since last contact (-1 = never) |
| `previous` | Numeric | Contacts before this campaign |
| `poutcome` | Categorical | Previous campaign outcome |

> ⚠️ **Note:** `duration` is highly predictive but only known after the call ends. In a real pre-call targeting scenario, this feature would not be available.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11 |
| ML Framework | scikit-learn, XGBoost |
| Imbalance Handling | imbalanced-learn (SMOTE) |
| Web App | Streamlit |
| Data Processing | pandas, NumPy |
| Visualization | matplotlib, seaborn |
| Model Serialization | joblib |

---

## 📁 Project Structure

```
project/
├── app/
│   └── main.py                # Streamlit application
├── data/
│   └── data.csv               # Bank marketing dataset
├── models/
│   ├── best_model.pkl         # Trained sklearn Pipeline
│   └── optimal_threshold.pkl  # Optimal probability threshold
├── src/
│   ├── __init__.py
│   ├── data_processing.py     # Data loading & preprocessing
│   ├── train.py               # Model training pipeline
│   └── evaluate.py            # Evaluation metrics & plots
├── notebooks/
│   ├── eda_analysis.py        # EDA script with visualizations
│   └── eda_plots/             # Generated EDA plots
├── .streamlit/
│   └── config.toml            # Streamlit theme configuration
├── requirements.txt
└── README.md
```

---

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/bank-marketing-prediction.git
cd bank-marketing-prediction
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the Model (if models/ is empty)

```bash
python -m src.train
```

### 5. Run the Streamlit App

```bash
streamlit run app/main.py
```

The app will open at `http://localhost:8501`

---

## 📈 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 88.9% |
| Precision | 51.8% |
| Recall | 74.9% |
| **F1-Score** | **61.2%** |
| **ROC-AUC** | **92.7%** |

*Evaluated on 20% holdout test set with optimal threshold of 0.15*

---

## 🖥️ Screenshots

### Manual Prediction
<!-- Add screenshot here -->
*Interactive form for single client prediction with real-time results*

### Bulk Prediction
<!-- Add screenshot here -->
*Upload files, validate columns, predict in bulk, and download results*

---

## 🔧 How It Works

1. **Preprocessing Pipeline** — StandardScaler for numeric features + OneHotEncoder for categorical features via ColumnTransformer
2. **Class Imbalance** — SMOTE oversampling on training data (88:12 → 50:50)
3. **Model Training** — Logistic Regression, Decision Tree, Random Forest, XGBoost compared
4. **Hyperparameter Tuning** — RandomizedSearchCV with 50 iterations, 5-fold stratified CV
5. **Threshold Optimization** — Custom threshold sweep (0.1-0.9) to maximize F1-Score
6. **Deployment Pipeline** — sklearn Pipeline wrapping preprocessor + classifier for raw DataFrame input

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/bank+marketing) for the dataset
- [Streamlit](https://streamlit.io/) for the web framework
- [scikit-learn](https://scikit-learn.org/) for the ML pipeline
