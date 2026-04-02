"""
Model Evaluation Module for Bank Marketing Prediction
======================================================
Functions for computing metrics, generating plots, and cross-validation.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)
from sklearn.model_selection import cross_val_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_metrics(y_true, y_pred, y_proba=None) -> dict:
    """
    Compute classification metrics.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    y_proba : array-like, optional
        Predicted probabilities for positive class.

    Returns
    -------
    dict
        Dictionary of metric names and values.
    """
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1-Score': f1_score(y_true, y_pred, zero_division=0),
    }

    if y_proba is not None:
        metrics['ROC-AUC'] = roc_auc_score(y_true, y_proba)

    logger.info(f"Metrics computed: {metrics}")
    return metrics


def classification_report_df(y_true, y_pred) -> pd.DataFrame:
    """
    Generate classification report as a DataFrame.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.

    Returns
    -------
    pd.DataFrame
        Classification report as a formatted DataFrame.
    """
    report = classification_report(y_true, y_pred, target_names=['No (0)', 'Yes (1)'], output_dict=True)
    df = pd.DataFrame(report).transpose()
    return df.round(4)


def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    """
    Plot a confusion matrix heatmap.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    title : str
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
        The confusion matrix figure.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No (0)', 'Yes (1)'],
                yticklabels=['No (0)', 'Yes (1)'], ax=ax)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_roc_curve(y_true, y_proba, title='ROC Curve'):
    """
    Plot ROC curve with AUC score.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_proba : array-like
        Predicted probabilities for positive class.
    title : str
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
        The ROC curve figure.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='#FF6B6B', linewidth=2, label=f'ROC Curve (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1, label='Random Classifier')
    ax.fill_between(fpr, tpr, alpha=0.1, color='#FF6B6B')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_feature_importance(model, feature_names, top_n=20, title='Top Feature Importances'):
    """
    Plot feature importance from a tree-based model.

    Parameters
    ----------
    model : estimator
        A fitted model with feature_importances_ attribute.
    feature_names : list
        List of feature names.
    top_n : int
        Number of top features to display.
    title : str
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
        The feature importance figure.
    """
    if not hasattr(model, 'feature_importances_'):
        logger.warning("Model does not have feature_importances_ attribute")
        return None

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    top_features = [feature_names[i] if i < len(feature_names) else f'Feature_{i}' for i in indices]
    top_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
    bars = ax.barh(range(len(top_features)), top_importances[::-1], color=colors)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features[::-1], fontsize=10)
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    return fig


def cross_validate_model(pipeline, X, y, cv=5) -> dict:
    """
    Perform cross-validation and return scores.

    Parameters
    ----------
    pipeline : sklearn Pipeline
        The full prediction pipeline.
    X : pd.DataFrame
        Feature data.
    y : pd.Series
        Target variable.
    cv : int
        Number of cross-validation folds.

    Returns
    -------
    dict
        Dictionary with metric names and lists of fold scores.
    """
    logger.info(f"Running {cv}-fold cross-validation")

    scoring_metrics = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc',
    }

    results = {}
    for name, scorer in scoring_metrics.items():
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scorer)
        results[name] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores.tolist()
        }
        logger.info(f"{name}: {scores.mean():.4f} (+/- {scores.std():.4f})")

    return results
