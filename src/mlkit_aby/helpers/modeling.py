from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
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