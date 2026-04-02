"""
Model Training Module for Bank Marketing Prediction
=====================================================
Handles model training, hyperparameter tuning, threshold optimization,
pipeline construction, and model persistence.
"""

import logging
import os
import sys
import numpy as np
import pandas as pd
import json
import skops.io as sio
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, roc_auc_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_processing import load_data, get_feature_target_split, build_preprocessor, get_feature_names
from src.evaluate import compute_metrics, classification_report_df, cross_validate_model

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_test_data(X, y, test_size=0.2, random_state=42):
    """
    Split data into train and test sets with stratification.

    Parameters
    ----------
    X : pd.DataFrame
        Feature data.
    y : pd.Series
        Target variable.
    test_size : float
        Proportion for test set.
    random_state : int
        Random seed.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    logger.info(f"Splitting data: test_size={test_size}, random_state={random_state}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    logger.info(f"Train target dist: {y_train.value_counts().to_dict()}")
    logger.info(f"Test target dist:  {y_test.value_counts().to_dict()}")
    return X_train, X_test, y_train, y_test


def get_models() -> dict:
    """
    Get a dictionary of models to train.

    Returns
    -------
    dict
        Model name -> model instance.
    """
    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=42, class_weight='balanced'
        ),
        'Decision Tree': DecisionTreeClassifier(
            random_state=42, class_weight='balanced'
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200, random_state=42, class_weight='balanced', n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=200, random_state=42, eval_metric='logloss',
            scale_pos_weight=7.5, use_label_encoder=False
        ),
    }
    return models


def train_and_evaluate_all(X_train, X_test, y_train, y_test, preprocessor):
    """
    Train all candidate models with SMOTE and evaluate them.

    Parameters
    ----------
    X_train, X_test : pd.DataFrame
        Train/test feature data.
    y_train, y_test : pd.Series
        Train/test target.
    preprocessor : ColumnTransformer
        Preprocessing pipeline.

    Returns
    -------
    tuple
        (results_df: pd.DataFrame, trained_models: dict)
    """
    models = get_models()
    results = []
    trained_models = {}

    # Preprocess training data and apply SMOTE
    logger.info("Fitting preprocessor and applying SMOTE on training data")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_processed, y_train)
    logger.info(f"After SMOTE - Train shape: {X_train_resampled.shape}, "
                f"Target dist: {pd.Series(y_train_resampled).value_counts().to_dict()}")

    for name, model in models.items():
        logger.info(f"\n{'='*50}\nTraining: {name}\n{'='*50}")

        # Train
        model.fit(X_train_resampled, y_train_resampled)

        # Predict
        y_pred = model.predict(X_test_processed)
        y_proba = model.predict_proba(X_test_processed)[:, 1] if hasattr(model, 'predict_proba') else None

        # Compute metrics
        metrics = compute_metrics(y_test, y_pred, y_proba)
        metrics['Model'] = name
        results.append(metrics)
        trained_models[name] = model

        logger.info(f"{name} Results: {metrics}")

    results_df = pd.DataFrame(results).set_index('Model')
    results_df = results_df[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']]

    logger.info(f"\n{'='*60}\nModel Comparison:\n{results_df.to_string()}\n{'='*60}")

    return results_df, trained_models


def tune_best_model(X_train, y_train, preprocessor, model_name='XGBoost'):
    """
    Perform hyperparameter tuning using RandomizedSearchCV.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training target.
    preprocessor : ColumnTransformer
        Preprocessing pipeline (must be already fitted).
    model_name : str
        Name of the model to tune.

    Returns
    -------
    tuple
        (best_model, best_params, best_score)
    """
    logger.info(f"Tuning hyperparameters for {model_name}")

    # Apply preprocessing + SMOTE
    X_train_processed = preprocessor.transform(X_train)
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train_processed, y_train)

    param_grids = {
        'XGBoost': {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [3, 5, 7, 9],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2],
            'scale_pos_weight': [1, 5, 7.5],
        },
        'Random Forest': {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2'],
            'class_weight': ['balanced', 'balanced_subsample'],
        },
        'Logistic Regression': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l2'],
            'class_weight': ['balanced'],
            'max_iter': [1000],
        },
        'Decision Tree': {
            'max_depth': [3, 5, 7, 10, 15, None],
            'min_samples_split': [2, 5, 10, 20],
            'min_samples_leaf': [1, 2, 4, 8],
            'class_weight': ['balanced'],
        },
    }

    if model_name not in param_grids:
        logger.error(f"Unknown model: {model_name}")
        return None, None, None

    base_models = get_models()
    model = base_models[model_name]
    param_grid = param_grids[model_name]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=50,
        scoring='f1',
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X_resampled, y_resampled)

    logger.info(f"Best params: {search.best_params_}")
    logger.info(f"Best F1 score (CV): {search.best_score_:.4f}")

    return search.best_estimator_, search.best_params_, search.best_score_


def find_optimal_threshold(model, X_test_processed, y_test):
    """
    Find the optimal probability threshold to maximize F1-score.

    Parameters
    ----------
    model : estimator
        Fitted model with predict_proba.
    X_test_processed : array-like
        Preprocessed test features.
    y_test : array-like
        True test labels.

    Returns
    -------
    tuple
        (optimal_threshold, best_f1, threshold_results_df)
    """
    logger.info("Finding optimal probability threshold")

    y_proba = model.predict_proba(X_test_processed)[:, 1]

    thresholds = np.arange(0.1, 0.91, 0.05)
    results = []

    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)
        results.append({
            'Threshold': round(thresh, 2),
            'F1-Score': round(f1, 4),
            'ROC-AUC': round(auc, 4),
        })

    results_df = pd.DataFrame(results)
    best_idx = results_df['F1-Score'].idxmax()
    optimal_threshold = results_df.loc[best_idx, 'Threshold']
    best_f1 = results_df.loc[best_idx, 'F1-Score']

    logger.info(f"Optimal threshold: {optimal_threshold} (F1: {best_f1})")

    return optimal_threshold, best_f1, results_df


def build_final_pipeline(preprocessor, model) -> Pipeline:
    """
    Build the final sklearn Pipeline combining preprocessing and model.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        The preprocessing transformer.
    model : estimator
        The trained model.

    Returns
    -------
    Pipeline
        Complete prediction pipeline.
    """
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model),
    ])
    return pipeline


def save_model(pipeline, threshold, models_dir='models'):
    """
    Save the trained pipeline and optimal threshold.

    Parameters
    ----------
    pipeline : Pipeline
        Trained prediction pipeline.
    threshold : float
        Optimal probability threshold.
    models_dir : str
        Directory to save model files.
    """
    os.makedirs(models_dir, exist_ok=True)

    pipeline_path = os.path.join(models_dir, 'best_model.skops')
    threshold_path = os.path.join(models_dir, 'optimal_threshold.json')

    sio.dump(pipeline, pipeline_path)

    with open(threshold_path, 'w') as f:
        json.dump({'threshold': float(threshold)}, f)

    logger.info(f"Pipeline saved to {pipeline_path}")
    logger.info(f"Threshold saved to {threshold_path}")


def main():
    """Main training pipeline execution."""
    print("\n" + "=" * 70)
    print("     BANK MARKETING - TERM DEPOSIT PREDICTION")
    print("     Full Training Pipeline")
    print("=" * 70 + "\n")

    # ── 1. Load Data ──────────────────────────────────────────
    print("[1/10] Loading Data...")
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'data.csv')
    df = load_data(data_path)
    print(f"   Dataset shape: {df.shape}")
    print(f"   Target distribution:\n{df['y'].value_counts().to_string()}\n")

    # ── 2. Feature/Target Split ────────────────────────────────
    print("[2/10] Splitting Features and Target...")
    X, y = get_feature_target_split(df)

    # ── 3. Train/Test Split ────────────────────────────────────
    print("[3/10] Train/Test Split (80/20, stratified)...")
    X_train, X_test, y_train, y_test = train_test_data(X, y)

    # ── 4. Build Preprocessor ──────────────────────────────────
    print("[4/10] Building Preprocessing Pipeline...")
    preprocessor = build_preprocessor()

    # ── 5. Train & Evaluate All Models ─────────────────────────
    print("\n[5/10] Training All Models (with SMOTE)...")
    results_df, trained_models = train_and_evaluate_all(
        X_train, X_test, y_train, y_test, preprocessor
    )
    print("\n--- Model Comparison Results ---")
    print(results_df.to_string())

    # ── 6. Select Best Model ───────────────────────────────────
    best_model_name = results_df['F1-Score'].idxmax()
    print(f"\n>>> Best Model (by F1-Score): {best_model_name}")

    # ── 7. Hyperparameter Tuning ───────────────────────────────
    print(f"\n[6/10] Tuning {best_model_name}...")
    best_model, best_params, best_cv_score = tune_best_model(
        X_train, y_train, preprocessor, best_model_name
    )
    print(f"   Best CV F1-Score: {best_cv_score:.4f}")
    print(f"   Best Parameters: {best_params}")

    # ── 8. Find Optimal Threshold ──────────────────────────────
    print("\n[7/10] Finding Optimal Probability Threshold...")
    X_test_processed = preprocessor.transform(X_test)
    optimal_threshold, best_f1, threshold_results = find_optimal_threshold(
        best_model, X_test_processed, y_test
    )
    print(f"   Optimal Threshold: {optimal_threshold}")
    print(f"   Best F1-Score at threshold: {best_f1}")
    print(f"\n   Threshold Analysis:\n{threshold_results.to_string(index=False)}")

    # ── 9. Final Evaluation ────────────────────────────────────
    print("\n[8/10] Final Model Evaluation...")
    y_proba_final = best_model.predict_proba(X_test_processed)[:, 1]
    y_pred_final = (y_proba_final >= optimal_threshold).astype(int)
    final_metrics = compute_metrics(y_test, y_pred_final, y_proba_final)
    print(f"   Final Metrics: {final_metrics}")

    report = classification_report_df(y_test, y_pred_final)
    print(f"\n   Classification Report:\n{report.to_string()}")

    # ── 10. Build & Save Pipeline ──────────────────────────────
    print("\n[9/10] Building Final Pipeline & Saving...")

    # Re-fit preprocessor + best model on full training data with SMOTE
    final_preprocessor = build_preprocessor()
    X_train_processed = final_preprocessor.fit_transform(X_train)
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train_processed, y_train)
    best_model.fit(X_resampled, y_resampled)

    # Build pipeline with fitted preprocessor
    final_pipeline = build_final_pipeline(final_preprocessor, best_model)

    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    save_model(final_pipeline, optimal_threshold, models_dir)

    # ── 11. Cross-Validation ───────────────────────────────────
    print("\n[10/10] Cross-Validation on Full Pipeline...")
    # For CV, build a fresh pipeline
    cv_pipeline = Pipeline([
        ('preprocessor', build_preprocessor()),
        ('classifier', best_model),
    ])
    cv_results = cross_validate_model(cv_pipeline, X, y, cv=5)
    print("   Cross-Validation Results:")
    for metric, vals in cv_results.items():
        print(f"   {metric}: {vals['mean']:.4f} (+/- {vals['std']:.4f})")

    print("\n" + "=" * 70)
    print("   TRAINING COMPLETE!")
    print(f"   Model: {best_model_name}")
    print(f"   Threshold: {optimal_threshold}")
    print(f"   Final F1: {best_f1}")
    print("   Files saved to models/")
    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()
