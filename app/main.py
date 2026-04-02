"""
Bank Term Deposit Prediction System
====================================
Streamlit application for predicting term deposit subscriptions.
Supports both single (manual) and bulk (file upload) predictions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import skops.io as sio
import json
import os
import io
import sys

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from src.data_processing import (
    ALL_FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    VALID_CATEGORIES, EXPECTED_COLUMNS, create_sample_data, validate_columns
)

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="Bank Term Deposit Prediction System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Custom CSS — Dark Theme Matching Reference Design
# ============================================================
st.markdown("""
<style>
    /* ── Global Dark Theme ─────────────────────────────── */
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }

    /* ── Main Header ───────────────────────────────────── */
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.06);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    .main-header h1 {
        color: #FAFAFA;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .main-header p {
        color: #9CA3AF;
        font-size: 0.95rem;
        margin: 0.3rem 0 0 0;
    }

    /* ── Section Headers ───────────────────────────────── */
    .section-header {
        font-size: 1.3rem;
        font-weight: 600;
        color: #FAFAFA;
        margin: 1.5rem 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* ── Info Card ──────────────────────────────────────── */
    .info-card {
        background: #1E1E2E;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1rem;
    }

    /* ── Result Cards ──────────────────────────────────── */
    .result-card {
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin: 0.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .result-yes {
        background: linear-gradient(135deg, #064e3b, #065f46);
        border-color: #10B981;
    }
    .result-no {
        background: linear-gradient(135deg, #7f1d1d, #991b1b);
        border-color: #EF4444;
    }
    .result-card h2 {
        margin: 0;
        font-size: 1.5rem;
        font-weight: 700;
    }
    .result-card p {
        margin: 0.3rem 0 0 0;
        font-size: 1rem;
        opacity: 0.85;
    }
    .result-card .probability {
        font-size: 2rem;
        font-weight: 800;
        margin-top: 0.5rem;
    }

    /* ── Predict Button ────────────────────────────────── */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #6366F1 0%, #8B5CF6 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-size: 1rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #4F46E5 0%, #7C3AED 100%);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5);
        transform: translateY(-1px);
    }

    /* ── Sidebar ──────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background-color: #0E1117;
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }
    section[data-testid="stSidebar"] .stRadio > label {
        color: #FAFAFA;
        font-weight: 600;
    }

    /* ── Download Buttons Row ─────────────────────────── */
    .download-btn-container {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }

    /* ── Input Fields Styling ─────────────────────────── */
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > div {
        background-color: #1E1E2E !important;
        color: #FAFAFA !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
    }

    /* ── Metric Cards ─────────────────────────────────── */
    .metric-row {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background: #1E1E2E;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 10px;
        padding: 1rem;
        flex: 1;
        text-align: center;
    }
    .metric-card .label {
        font-size: 0.8rem;
        color: #9CA3AF;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-card .value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #FAFAFA;
        margin-top: 0.3rem;
    }

    /* ── Tab / Divider styling ────────────────────────── */
    .gradient-divider {
        height: 3px;
        background: linear-gradient(90deg, #6366F1, #8B5CF6, #A855F7, #6366F1);
        border: none;
        border-radius: 2px;
        margin: 0.5rem 0 1.5rem 0;
    }

    /* ── File Upload Area ─────────────────────────────── */
    .stFileUploader > div {
        background-color: #1E1E2E !important;
        border: 1px dashed rgba(255, 255, 255, 0.15) !important;
        border-radius: 10px !important;
    }

    /* ── DataFrame ────────────────────────────────────── */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ============================================================
# Model Loading (Cached)
# ============================================================
@st.cache_resource
def load_model():
    """Load the trained pipeline and optimal threshold."""
    model_path = os.path.join(PROJECT_ROOT, 'models', 'best_model.skops')
    threshold_path = os.path.join(PROJECT_ROOT, 'models', 'optimal_threshold.json')

    if not os.path.exists(model_path):
        st.error("Model file not found! Please run `python -m src.train` first.")
        st.stop()

    unknown_types = sio.get_untrusted_types(file=model_path)
    pipeline = sio.load(model_path, trusted=unknown_types)

    if os.path.exists(threshold_path):
        with open(threshold_path, 'r') as f:
            threshold = json.load(f)['threshold']
    else:
        threshold = 0.5

    return pipeline, threshold


# ============================================================
# Header
# ============================================================
st.markdown("""
<div class="main-header">
    <h1>🏦 Bank Term Deposit Prediction System</h1>
    <p>Predict whether a client will subscribe to a term deposit based on their profile and campaign history.</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="gradient-divider"></div>', unsafe_allow_html=True)

# ============================================================
# Sidebar Navigation
# ============================================================
with st.sidebar:
    st.markdown("## 🧭 Navigation")
    page = st.radio(
        "Select Prediction Mode",
        ["🎯 Manual Prediction", "🔍 Bulk Prediction"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("""
    <div class="info-card">
        <strong>About</strong><br>
        <span style="color: #9CA3AF; font-size: 0.85rem;">
        This system uses an ML pipeline trained on the Bank Marketing dataset 
        to predict term deposit subscriptions. It leverages XGBoost with SMOTE 
        for handling class imbalance and custom threshold tuning for optimal F1-score.
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-card">
        <strong>Model Info</strong><br>
        <span style="color: #9CA3AF; font-size: 0.85rem;">
        • Algorithm: XGBoost (tuned)<br>
        • Preprocessing: StandardScaler + OneHotEncoder<br>
        • Imbalance: SMOTE oversampling<br>
        • Threshold: Optimized for F1-Score
        </span>
    </div>
    """, unsafe_allow_html=True)


# Load model
pipeline, threshold = load_model()


# ============================================================
# MANUAL PREDICTION PAGE
# ============================================================
if page == "🎯 Manual Prediction":

    st.markdown('<div class="section-header">📋 Client Information</div>', unsafe_allow_html=True)

    # ── Row 1: Age | Housing | Duration ────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", min_value=18, max_value=95, value=30, step=1)
    with col2:
        housing = st.selectbox("Has Housing Loan?", VALID_CATEGORIES['housing'], index=0)
    with col3:
        duration = st.number_input("Last Contact Duration (seconds)", min_value=0, max_value=5000, value=200, step=10)

    # ── Row 2: Job | Loan | Campaign ──────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        job = st.selectbox("Job", VALID_CATEGORIES['job'], index=0)
    with col2:
        loan = st.selectbox("Has Personal Loan?", VALID_CATEGORIES['loan'], index=0)
    with col3:
        campaign = st.number_input("Number of Contacts during this campaign", min_value=1, max_value=63, value=1, step=1)

    # ── Row 3: Marital | Contact | Pdays ──────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        marital = st.selectbox("Marital Status", VALID_CATEGORIES['marital'], index=0)
    with col2:
        contact = st.selectbox("Contact Communication Type", VALID_CATEGORIES['contact'], index=0)
    with col3:
        pdays = st.number_input("Days since last contact (from previous campaign)", min_value=-1, max_value=871, value=-1, step=1,
                                help="-1 means client was not previously contacted")

    # ── Row 4: Education | Day | Previous ─────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        education = st.selectbox("Education", VALID_CATEGORIES['education'], index=0)
    with col2:
        day = st.number_input("Last Contact Day of the Month", min_value=1, max_value=31, value=15, step=1)
    with col3:
        previous = st.number_input("Number of Contacts performed before this campaign", min_value=0, max_value=275, value=0, step=1)

    # ── Row 5: Default | Month | Poutcome ─────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        default = st.selectbox("Has Credit in Default?", VALID_CATEGORIES['default'], index=0)
    with col2:
        month = st.selectbox("Last Contact Month", VALID_CATEGORIES['month'], index=0)
    with col3:
        poutcome = st.selectbox("Outcome of previous campaign", VALID_CATEGORIES['poutcome'], index=0)

    # ── Row 6: Balance ────────────────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        balance = st.number_input("Yearly Average Balance (in EUR)", min_value=-8019, max_value=102127, value=1000, step=100)

    st.markdown("---")

    # ── Predict Button ────────────────────────────────────
    if st.button("🚀 Predict Conversion", use_container_width=True):
        # Build input DataFrame
        input_data = pd.DataFrame([{
            'age': age, 'job': job, 'marital': marital, 'education': education,
            'default': default, 'balance': balance, 'housing': housing, 'loan': loan,
            'contact': contact, 'day': day, 'month': month, 'duration': duration,
            'campaign': campaign, 'pdays': pdays, 'previous': previous, 'poutcome': poutcome,
        }])

        # Predict
        try:
            proba = pipeline.predict_proba(input_data)[:, 1][0]
            prediction = "Yes" if proba >= threshold else "No"

            st.markdown("---")
            st.markdown('<div class="section-header">📊 Prediction Result</div>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                if prediction == "Yes":
                    st.markdown(f"""
                    <div class="result-card result-yes">
                        <h2>✅ Will Subscribe</h2>
                        <p>The client is <strong>likely</strong> to subscribe to a term deposit.</p>
                        <div class="probability">{proba:.1%}</div>
                        <p>Probability</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-card result-no">
                        <h2>❌ Will Not Subscribe</h2>
                        <p>The client is <strong>unlikely</strong> to subscribe to a term deposit.</p>
                        <div class="probability">{proba:.1%}</div>
                        <p>Probability</p>
                    </div>
                    """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="result-card" style="background: #1E1E2E;">
                    <h2>📈 Prediction Details</h2>
                    <br>
                    <div style="text-align: left; padding: 0 1rem;">
                        <p><strong>Prediction:</strong> {prediction}</p>
                        <p><strong>Probability:</strong> {proba:.4f}</p>
                        <p><strong>Threshold Used:</strong> {threshold}</p>
                        <p><strong>Confidence:</strong> {'High' if abs(proba - 0.5) > 0.3 else 'Medium' if abs(proba - 0.5) > 0.15 else 'Low'}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")


# ============================================================
# BULK PREDICTION PAGE
# ============================================================
elif page == "🔍 Bulk Prediction":

    st.markdown('<div class="section-header">🔍 Bulk Prediction Scanner</div>', unsafe_allow_html=True)

    # ── 1. Sample Templates ───────────────────────────────
    st.markdown("### 1. Download Sample Templates 🔗")
    st.markdown("Download a template file with the correct column structure:")

    sample_df = create_sample_data()

    col1, col2, col3 = st.columns(3)

    with col1:
        csv_data = sample_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📄 Download CSV Sample",
            data=csv_data,
            file_name="sample_template.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col2:
        excel_buffer = io.BytesIO()
        sample_df.to_excel(excel_buffer, index=False, engine='openpyxl')
        excel_data = excel_buffer.getvalue()
        st.download_button(
            label="📗 Download Excel Sample",
            data=excel_data,
            file_name="sample_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    with col3:
        json_data = sample_df.to_json(orient='records', indent=2).encode('utf-8')
        st.download_button(
            label="🟠 Download JSON Sample",
            data=json_data,
            file_name="sample_template.json",
            mime="application/json",
            use_container_width=True,
        )

    st.markdown("---")

    # ── 2. File Upload ────────────────────────────────────
    st.markdown("### 2. Upload File to Scan")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["csv", "xlsx", "json"],
        help="Upload a CSV, Excel (.xlsx), or JSON file with the correct column structure. Limit 200MB per file."
    )

    if uploaded_file is not None:
        # Parse the uploaded file
        try:
            file_ext = uploaded_file.name.split('.')[-1].lower()

            if file_ext == 'csv':
                df_upload = pd.read_csv(uploaded_file)
            elif file_ext in ['xlsx', 'xls']:
                df_upload = pd.read_excel(uploaded_file, engine='openpyxl')
            elif file_ext == 'json':
                df_upload = pd.read_json(uploaded_file)
            else:
                st.error(f"Unsupported file format: .{file_ext}")
                st.stop()

            # Normalize column names
            df_upload.columns = df_upload.columns.str.strip().str.lower()

            st.success(f"File loaded successfully! **{df_upload.shape[0]} rows** and **{df_upload.shape[1]} columns** detected.")

            # ── 3. Validation ─────────────────────────────
            is_valid, missing_cols, extra_cols = validate_columns(df_upload)

            if not is_valid:
                st.error(f"**Column Validation Failed!** Missing required columns: `{', '.join(missing_cols)}`")
                st.markdown("**Expected columns:**")
                st.code(', '.join(EXPECTED_COLUMNS))
                st.markdown("**Your columns:**")
                st.code(', '.join(df_upload.columns.tolist()))
                st.stop()

            if extra_cols:
                st.warning(f"Extra columns found and will be ignored: `{', '.join(extra_cols)}`")

            # ── 4. Preview ────────────────────────────────
            st.markdown("### 3. Data Preview")
            st.dataframe(df_upload.head(5), use_container_width=True)

            st.markdown(f"""
            <div class="metric-row">
                <div class="metric-card">
                    <div class="label">Total Rows</div>
                    <div class="value">{df_upload.shape[0]:,}</div>
                </div>
                <div class="metric-card">
                    <div class="label">Total Columns</div>
                    <div class="value">{df_upload.shape[1]}</div>
                </div>
                <div class="metric-card">
                    <div class="label">Missing Values</div>
                    <div class="value">{df_upload[EXPECTED_COLUMNS].isnull().sum().sum()}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")

            # ── 5. Bulk Prediction ────────────────────────
            st.markdown("### 4. Run Predictions")

            if st.button("🚀 Predict All Rows", use_container_width=True):
                with st.spinner("Running predictions..."):
                    progress_bar = st.progress(0, text="Preparing data...")

                    # Prepare feature data
                    df_features = df_upload[EXPECTED_COLUMNS].copy()
                    progress_bar.progress(20, text="Preprocessing features...")

                    try:
                        # Get predictions
                        probabilities = pipeline.predict_proba(df_features)[:, 1]
                        progress_bar.progress(70, text="Generating predictions...")

                        predictions = ["Yes" if p >= threshold else "No" for p in probabilities]
                        progress_bar.progress(90, text="Assembling results...")

                        # Add results to dataframe
                        df_results = df_upload.copy()
                        df_results['Prediction'] = predictions
                        df_results['Probability'] = [round(p, 4) for p in probabilities]
                        progress_bar.progress(100, text="Complete!")

                        # ── 6. Results Display ────────────
                        st.markdown("### 5. Prediction Results")

                        # Summary metrics
                        yes_count = predictions.count("Yes")
                        no_count = predictions.count("No")
                        total = len(predictions)
                        avg_prob = np.mean(probabilities)

                        st.markdown(f"""
                        <div class="metric-row">
                            <div class="metric-card">
                                <div class="label">Total Predictions</div>
                                <div class="value">{total:,}</div>
                            </div>
                            <div class="metric-card" style="border-color: #10B981;">
                                <div class="label">Will Subscribe (Yes)</div>
                                <div class="value" style="color: #10B981;">{yes_count:,} ({yes_count/total:.1%})</div>
                            </div>
                            <div class="metric-card" style="border-color: #EF4444;">
                                <div class="label">Will Not Subscribe (No)</div>
                                <div class="value" style="color: #EF4444;">{no_count:,} ({no_count/total:.1%})</div>
                            </div>
                            <div class="metric-card">
                                <div class="label">Avg. Probability</div>
                                <div class="value">{avg_prob:.4f}</div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Show full results table
                        st.dataframe(
                            df_results.style.apply(
                                lambda row: ['background-color: rgba(16, 185, 129, 0.1)' if row['Prediction'] == 'Yes'
                                             else 'background-color: rgba(239, 68, 68, 0.1)'] * len(row),
                                axis=1
                            ),
                            use_container_width=True,
                            height=400,
                        )

                        # ── 7. Download Results ───────────
                        st.markdown("### 6. Download Results")

                        csv_results = df_results.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="📥 Download Predictions (CSV)",
                            data=csv_results,
                            file_name="predictions_result.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )

                    except Exception as e:
                        progress_bar.empty()
                        st.error(f"Prediction failed: {str(e)}")
                        st.markdown("""
                        **Possible causes:**
                        - Data contains values the model hasn't seen before
                        - Missing values in required columns
                        - Data types don't match expected schema
                        
                        Please check your file and try again.
                        """)

        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
            st.markdown("Please ensure the file is not corrupted and is in the correct format.")
