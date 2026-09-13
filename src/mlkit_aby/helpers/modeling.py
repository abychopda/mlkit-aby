import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV

def run_logistic_regression_grid_search(
    X_train,
    y_train,
    param_grid,
    cv=5,
    scoring="average_precision",
    random_state=42,
):
    """
    Run cross-validated hyperparameter search for Logistic Regression.

    Returns:
        grid_search: Fitted GridSearchCV object.
    """
    model = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        max_iter=2000,
        random_state=random_state,
    )

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        refit=True,
    )

    grid_search.fit(X_train, y_train)

    return grid_search


def train_logistic_regression(
    X_train,
    y_train,
    C=0.1,
    class_weight=None,
    penalty="l2",
    solver="lbfgs",
    max_iter=2000,
    random_state=42,
):
    """
    Train a Logistic Regression model using the supplied hyperparameters.

    Returns:
        model: Fitted LogisticRegression model.
    """
    model = LogisticRegression(
        C=C,
        class_weight=class_weight,
        penalty=penalty,
        solver=solver,
        max_iter=max_iter,
        random_state=random_state,
    )

    model.fit(X_train, y_train)

    return model


def evaluate_logistic_regression_auprc(
    model,
    X,
    y,
):
    """
    Evaluate a fitted Logistic Regression model using AUPRC.

    Returns:
        float: Average Precision score.
    """
    predicted_probabilities = model.predict_proba(X)[:, 1]

    return average_precision_score(
        y,
        predicted_probabilities,
    )

# --------------------------------------------------------------------------
# Regression
# --------------------------------------------------------------------------


def train_linear_regression(X_train, y_train):
    """
    Train an ordinary least squares linear regression model.

    Args:
        X_train: Training features.
        y_train: Training target values.

    Returns:
        model: Fitted LinearRegression model.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)

    return model


def train_ridge_regression(X_train, y_train, alpha=1.0, random_state=42):
    """
    Train a ridge regression model.

    Ridge adds an L2 penalty on the coefficients, which stabilises the fit
    when features are correlated. Lagged and rolling weather features are
    strongly correlated with one another, so this is usually a better
    behaved choice than ordinary least squares.

    Args:
        X_train: Training features.
        y_train: Training target values.
        alpha: Strength of the L2 penalty. Larger values shrink harder.
        random_state: Seed used for reproducibility.

    Returns:
        model: Fitted Ridge model.
    """
    model = Ridge(alpha=alpha, random_state=random_state)
    model.fit(X_train, y_train)

    return model


def train_random_forest_regressor(
    X_train,
    y_train,
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=1,
    random_state=42,
):
    """
    Train a random forest regression model.

    Args:
        X_train: Training features.
        y_train: Training target values.
        n_estimators: Number of trees in the forest.
        max_depth: Maximum tree depth, or None for unlimited.
        min_samples_leaf: Minimum samples required at a leaf node.
        random_state: Seed used for reproducibility.

    Returns:
        model: Fitted RandomForestRegressor model.
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    return model


def evaluate_regression(y_true, y_pred) -> dict[str, float]:
    """
    Calculate standard regression performance metrics.

    Args:
        y_true: Observed target values.
        y_pred: Predicted target values.

    Returns:
        dict[str, float]:
            ``rmse`` and ``mae``, both in the units of the target, and
            ``r2``, the proportion of variance explained.
    """
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


# --------------------------------------------------------------------------
# Multiclass classification
# --------------------------------------------------------------------------


def train_random_forest_classifier(
    X_train,
    y_train,
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=42,
):
    """
    Train a random forest classifier.

    ``class_weight`` defaults to ``"balanced"`` because the hazard classes
    are severely imbalanced. Without it, a model can score well simply by
    never predicting the rare classes, which are the ones that matter most.

    Args:
        X_train: Training features.
        y_train: Training class labels.
        n_estimators: Number of trees in the forest.
        max_depth: Maximum tree depth, or None for unlimited.
        min_samples_leaf: Minimum samples required at a leaf node.
        class_weight: Class weighting strategy, or None for no weighting.
        random_state: Seed used for reproducibility.

    Returns:
        model: Fitted RandomForestClassifier model.
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    return model


def evaluate_classification(
    y_true,
    y_pred,
    labels=None,
) -> dict[str, float]:
    """
    Calculate performance metrics for a multiclass classification model.

    Accuracy is reported for completeness but should not be used as the
    headline metric on imbalanced data: a model that always predicts the
    majority class can post a high accuracy while being useless. Macro F1
    and balanced accuracy weight every class equally and expose that
    failure.

    Args:
        y_true: Observed class labels.
        y_pred: Predicted class labels.
        labels: Full list of possible class labels. Supplying this ensures
            classes absent from a given split are still reported, rather
            than silently omitted.

    Returns:
        dict[str, float]:
            Overall accuracy, balanced accuracy, macro and weighted F1,
            and per-class recall and F1 keyed by class label.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)),
    }

    per_class_recall = recall_score(
        y_true, y_pred, average=None, labels=labels, zero_division=0
    )
    per_class_f1 = f1_score(
        y_true, y_pred, average=None, labels=labels, zero_division=0
    )

    for label, recall, f1 in zip(labels, per_class_recall, per_class_f1):
        metrics[f"recall_class_{label}"] = float(recall)
        metrics[f"f1_class_{label}"] = float(f1)

    return metrics
