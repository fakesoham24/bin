"""
Data Processing Module for Bank Marketing Prediction
=====================================================
Handles data loading, feature engineering, and preprocessing pipeline construction.
"""

import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================
# Feature Definitions
# ============================================================

NUMERIC_FEATURES = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
CATEGORICAL_FEATURES = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Expected columns for validation (bulk prediction)
EXPECTED_COLUMNS = ALL_FEATURES.copy()

# Valid categories for each categorical feature (for UI dropdowns & validation)
VALID_CATEGORIES = {
    'job': ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
            'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'],
    'marital': ['divorced', 'married', 'single'],
    'education': ['primary', 'secondary', 'tertiary', 'unknown'],
    'default': ['no', 'yes'],
    'housing': ['no', 'yes'],
    'loan': ['no', 'yes'],
    'contact': ['cellular', 'telephone', 'unknown'],
    'month': ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
    'poutcome': ['failure', 'other', 'success', 'unknown'],
}


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the bank marketing dataset from a CSV file.

    Parameters
    ----------
    filepath : str
        Path to the CSV file (semicolon-separated).

    Returns
    -------
    pd.DataFrame
        Loaded dataframe.
    """
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath, sep=';')
    logger.info(f"Data loaded successfully. Shape: {df.shape}")
    return df


def get_feature_target_split(df: pd.DataFrame):
    """
    Split dataframe into features (X) and encoded target (y).

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe containing all columns including 'y'.

    Returns
    -------
    tuple
        (X, y) where X is a DataFrame of features and y is a Series of 0/1.
    """
    logger.info("Splitting features and target")
    X = df[ALL_FEATURES].copy()
    y = df['y'].map({'no': 0, 'yes': 1})
    logger.info(f"Features shape: {X.shape}, Target distribution:\n{y.value_counts().to_dict()}")
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """
    Build a ColumnTransformer that applies:
    - StandardScaler to numeric features
    - OneHotEncoder to categorical features

    Returns
    -------
    ColumnTransformer
        Configured preprocessing transformer.
    """
    logger.info("Building preprocessing pipeline")

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERIC_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )

    logger.info("Preprocessor built successfully")
    return preprocessor


def get_feature_names(preprocessor, fitted=True) -> list:
    """
    Get feature names from a fitted ColumnTransformer.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        Fitted preprocessor.
    fitted : bool
        Whether the preprocessor has been fitted.

    Returns
    -------
    list
        List of feature names after transformation.
    """
    if not fitted:
        return []
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        return NUMERIC_FEATURES + ['encoded_feature']


def validate_columns(df: pd.DataFrame) -> tuple:
    """
    Validate that a DataFrame has the expected columns for prediction.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.

    Returns
    -------
    tuple
        (is_valid: bool, missing_cols: list, extra_cols: list)
    """
    df_cols = set(df.columns.str.strip().str.lower())
    expected = set(c.lower() for c in EXPECTED_COLUMNS)

    missing = expected - df_cols
    extra = df_cols - expected

    is_valid = len(missing) == 0
    return is_valid, sorted(missing), sorted(extra)


def create_sample_data() -> pd.DataFrame:
    """
    Create a sample DataFrame with correct schema for bulk prediction templates.

    Returns
    -------
    pd.DataFrame
        Sample data with 5 rows matching the expected schema.
    """
    sample = pd.DataFrame({
        'age': [30, 45, 35, 50, 28],
        'job': ['admin.', 'management', 'technician', 'retired', 'student'],
        'marital': ['married', 'single', 'divorced', 'married', 'single'],
        'education': ['secondary', 'tertiary', 'secondary', 'primary', 'tertiary'],
        'default': ['no', 'no', 'no', 'no', 'no'],
        'balance': [1500, 3000, 500, 8000, 200],
        'housing': ['yes', 'no', 'yes', 'no', 'no'],
        'loan': ['no', 'no', 'yes', 'no', 'no'],
        'contact': ['cellular', 'telephone', 'cellular', 'cellular', 'unknown'],
        'day': [15, 20, 5, 10, 25],
        'month': ['jan', 'mar', 'jun', 'oct', 'dec'],
        'duration': [200, 350, 100, 500, 150],
        'campaign': [1, 2, 3, 1, 4],
        'pdays': [-1, 100, -1, 200, -1],
        'previous': [0, 2, 0, 3, 0],
        'poutcome': ['unknown', 'success', 'unknown', 'failure', 'unknown'],
    })
    return sample
