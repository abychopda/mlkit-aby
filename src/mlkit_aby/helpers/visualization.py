"""
Data visualisation utility functions.

This module contains reusable functions for generating visualisations used
during exploratory data analysis and model interpretation. Functions include
plots for feature distributions, correlations, missing values, class
distributions, and other analytical visualisations.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
)


def plot_numeric_feature_by_target(
    df: pd.DataFrame,
    feature_name: str,
    target_name: str,
    display_name: str,
    bins: int = 30,
) -> Figure:
    """Create distribution and box plots for a numeric feature by target class.

    Args:
        df: Dataset containing the feature and target columns.
        feature_name: Name of the numeric feature to plot.
        target_name: Name of the target column that defines the groups.
        display_name: Human-readable feature name for plot labels.
        bins: Number of histogram bins. Defaults to 30.

    Returns:
        A Matplotlib figure containing a class-wise histogram and box plot.

    Raises:
        KeyError: If the feature or target column is not present in the dataset.
        ValueError: If bins is less than 1.
    """
    required_columns = {feature_name, target_name}
    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise KeyError(f"Missing required columns: {sorted(missing_columns)}")
    if bins < 1:
        raise ValueError("bins must be at least 1")

    plot_data = df[[feature_name, target_name]].dropna()
    figure, axes = plt.subplots(ncols=2, figsize=(13, 4))

    sns.histplot(
        data=plot_data,
        x=feature_name,
        hue=target_name,
        bins=bins,
        stat="density",
        common_norm=False,
        element="step",
        ax=axes[0],
    )
    axes[0].set(title=f"{display_name}: distribution by label")

    sns.boxplot(data=plot_data, x=target_name, y=feature_name, ax=axes[1])
    axes[1].set(title=f"{display_name}: class comparison", xlabel=target_name)

    figure.tight_layout()
    return figure

def plot_correlation_heatmap(
    correlation_matrix: pd.DataFrame,
    figsize: tuple[int, int] = (16, 12),
    title: str = "Correlation Matrix",
) -> None:
    """
    Plot a heatmap of a correlation matrix.

    This function creates a reusable heatmap for visualising the strength
    and direction of relationships between numerical features.

    Args:
        correlation_matrix (pd.DataFrame):
            A square pandas DataFrame containing pairwise correlation
            coefficients between numerical features.

        figsize (tuple[int, int], optional):
            Width and height of the figure in inches.
            Defaults to (16, 12).

        title (str, optional):
            Title displayed above the heatmap.
            Defaults to "Correlation Matrix".

    Returns:
        None:
            Displays the correlation heatmap.

    Raises:
        ValueError:
            If the supplied correlation matrix is empty or is not square.
    """
    if correlation_matrix.empty:
        raise ValueError("The correlation matrix cannot be empty.")

    if correlation_matrix.shape[0] != correlation_matrix.shape[1]:
        raise ValueError("The correlation matrix must be square.")

    plt.figure(figsize=figsize)

    sns.heatmap(
        correlation_matrix,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
    )

    plt.title(title)
    plt.tight_layout()
    plt.show()



def plot_precision_recall_curve(
    y_true,
    y_proba,
    title="Precision-Recall Curve",
):
    """
    Plot the Precision-Recall curve and display the AUPRC.

    Args:
        y_true: True binary target values.
        y_proba: Predicted probabilities for the positive class.
        title: Title for the plot.

    Returns:
        float: AUPRC (average precision) score.
    """
    precision, recall, _ = precision_recall_curve(
        y_true,
        y_proba,
    )

    auprc = average_precision_score(
        y_true,
        y_proba,
    )

    plt.figure(figsize=(8, 6))

    plt.plot(
        recall,
        precision,
        label=f"AUPRC = {auprc:.4f}",
    )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

    return auprc


def plot_logistic_coefficients(
    model,
    feature_names,
    n_features=10,
    title="Logistic Regression Feature Coefficients",
):
    """
    Plot the strongest positive and negative Logistic Regression
    coefficients.

    Args:
        model: Fitted Logistic Regression model.
        feature_names: Names of the model input features.
        n_features: Number of strongest positive and negative
            coefficients to display.
        title: Title for the plot.

    Returns:
        DataFrame containing the plotted coefficients.
    """
    coefficients = pd.Series(
        model.coef_[0],
        index=feature_names,
    )

    strongest_positive = coefficients.nlargest(n_features)
    strongest_negative = coefficients.nsmallest(n_features)

    selected_coefficients = pd.concat(
        [strongest_negative, strongest_positive]
    ).sort_values()

    plt.figure(figsize=(10, 8))

    selected_coefficients.plot(
        kind="barh"
    )

    plt.xlabel("Logistic Regression Coefficient")
    plt.ylabel("Feature")
    plt.title(title)
    plt.grid(True, axis="x")
    plt.tight_layout()
    plt.show()

    return selected_coefficients.to_frame(
        name="coefficient"
    )




def plot_confusion_matrix(
    y_true,
    y_proba,
    threshold=0.5,
    title="Confusion Matrix",
):
    """
    Plot a confusion matrix using a specified probability threshold.

    Args:
        y_true: True binary target values.
        y_proba: Predicted probabilities for the positive class.
        threshold: Probability threshold used to convert probabilities
            into binary predictions.
        title: Title for the plot.

    Returns:
        DataFrame containing the confusion matrix counts.
    """
    # Convert predicted probabilities into binary predictions
    # using the specified classification threshold.
    y_pred = (y_proba >= threshold).astype(int)

    # Calculate the four confusion-matrix outcomes.
    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    confusion_table = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"],
    )

    plt.figure(figsize=(7, 6))

    plt.imshow(confusion_table.values, cmap="Blues",)

    plt.xticks(
        [0, 1],
        confusion_table.columns,
    )

    plt.yticks(
        [0, 1],
        confusion_table.index,
    )

    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.title(
        f"{title}\nThreshold = {threshold:.2f}"
    )

    # Display the counts inside each cell.
    for i in range(2):
        for j in range(2):
            value = confusion_table.iloc[i, j]

            text_color = "white" if value > confusion_table.values.max() * 0.5 else "black"

            plt.text(
                j,
                i,
                value,
                ha="center",
                va="center",
                color=text_color,
                fontsize=14,
            )

    plt.tight_layout()
    plt.show()

    return confusion_table

# --------------------------------------------------------------------------
# Regression diagnostics
# --------------------------------------------------------------------------


def plot_predicted_vs_actual(
    y_true,
    y_pred,
    title="Predicted vs Actual",
    figsize=(7, 7),
) -> Figure:
    """
    Plot predicted values against observed values.

    A perfect model places every point on the diagonal. Systematic
    departures reveal bias: points below the line at high values, for
    instance, mean the model under-predicts when the true value is large.

    Args:
        y_true: Observed target values.
        y_pred: Predicted target values.
        title: Title for the plot.
        figsize: Figure size in inches.

    Returns:
        Figure: The matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(y_true, y_pred, alpha=0.3, edgecolor="none")

    # Reference line showing where perfect predictions would fall.
    lower = min(np.min(y_true), np.min(y_pred))
    upper = max(np.max(y_true), np.max(y_pred))
    ax.plot([lower, upper], [lower, upper], linestyle="--", linewidth=1.5, color="black")

    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    fig.tight_layout()

    return fig


def plot_residuals(
    y_true,
    y_pred,
    title="Residuals vs Predicted",
    figsize=(9, 5),
) -> Figure:
    """
    Plot residuals against predicted values.

    Residuals should scatter randomly around zero with constant spread.
    A funnel shape indicates the error grows with the prediction, and
    curvature indicates the model is missing a non-linear relationship.

    Args:
        y_true: Observed target values.
        y_pred: Predicted target values.
        title: Title for the plot.
        figsize: Figure size in inches.

    Returns:
        Figure: The matplotlib figure.
    """
    residuals = np.asarray(y_true) - np.asarray(y_pred)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(y_pred, residuals, alpha=0.3, edgecolor="none")
    ax.axhline(0, linestyle="--", linewidth=1.5, color="black")

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (actual - predicted)")
    ax.set_title(title)
    fig.tight_layout()

    return fig


def plot_forecast_timeline(
    dates,
    y_true,
    y_pred,
    title="Forecast vs Actual Over Time",
    figsize=(14, 5),
) -> Figure:
    """
    Plot observed and predicted values against time.

    This is the view that reveals whether a forecast tracks seasonal
    structure or merely predicts the average, which the scatter plots
    cannot show.

    Args:
        dates: Dates corresponding to each observation.
        y_true: Observed target values.
        y_pred: Predicted target values.
        title: Title for the plot.
        figsize: Figure size in inches.

    Returns:
        Figure: The matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(dates, y_true, label="Actual", linewidth=1.2)
    ax.plot(dates, y_pred, label="Predicted", linewidth=1.2, alpha=0.8)

    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    return fig


# --------------------------------------------------------------------------
# Feature importance
# --------------------------------------------------------------------------


def plot_feature_importance(
    importance_df: pd.DataFrame,
    value_column="importance",
    feature_column="feature",
    top_n=20,
    title="Feature Importance",
    figsize=(9, 8),
) -> Figure:
    """
    Plot the most important features as a horizontal bar chart.

    Args:
        importance_df: Table of feature importances.
        value_column: Column holding the importance value.
        feature_column: Column holding the feature name.
        top_n: Number of features to display.
        title: Title for the plot.
        figsize: Figure size in inches.

    Returns:
        Figure: The matplotlib figure.
    """
    top_features = importance_df.nlargest(top_n, value_column).iloc[::-1]

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(top_features[feature_column], top_features[value_column])

    ax.set_xlabel(value_column.replace("_", " ").title())
    ax.set_title(title)
    fig.tight_layout()

    return fig


# --------------------------------------------------------------------------
# Multiclass classification diagnostics
# --------------------------------------------------------------------------


def plot_multiclass_confusion_matrix(
    y_true,
    y_pred,
    labels,
    display_labels=None,
    normalize=None,
    title="Confusion Matrix",
    figsize=(7, 6),
) -> Figure:
    """
    Plot a confusion matrix for a multiclass problem.

    Args:
        y_true: Observed class labels.
        y_pred: Predicted class labels.
        labels: Full list of possible class labels, in display order.
        display_labels: Human-readable names for each class.
        normalize: Passed to sklearn; use ``"true"`` to show per-class
            recall, which is easier to read when classes are imbalanced.
        title: Title for the plot.
        figsize: Figure size in inches.

    Returns:
        Figure: The matplotlib figure.
    """
    matrix = confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)

    if display_labels is None:
        display_labels = labels

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=display_labels,
        yticklabels=display_labels,
        ax=ax,
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()

    return fig
