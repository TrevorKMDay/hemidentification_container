import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def fit_model(cx, y, test=None):

    """Fit a linear discriminant model.

    Fit a model predicting y from the predictor data frame cx.
    If test=True, predict the predictor data frame against the same model and
    return the scores. Returns C-1 LDs, where *C* is the number of outcome
    categories.

    Keyword arguments:
    cx: The NxM matrix of predictors (M).
    y: A length-M vector of outcomes (categorical)
    test: Whether to test cx against the newly trained model (default: False).

    Returns a list of the following:
    clf: The model object.
    scalings: The scalings from the model.
    ex_var: The clf.explained_variance_ratio_ object
    y_hat: The predicted categories (None if test=False)
    scores: A Nx(C-1) matrix of LD scores (None if test=False)

    """

    # Fit the model
    clf = LinearDiscriminantAnalysis()
    clf.fit(cx, y)

    # Extract results from the model
    ex_var = clf.explained_variance_ratio_

    scalings = pd.DataFrame(clf.scalings_)
    scalings.columns = [f"LD{x}" for x in range(1, scalings.shape[1] + 1)]
    scalings["feature"] = clf.feature_names_in_

    if test is not None:

        y_hat = clf.predict(test)

        # Return pretty scores
        scores = pd.DataFrame(clf.transform(test))
        scores.columns = [f"LD{x}" for x in range(1, scores.shape[1] + 1)]
        scores = scores.reset_index(drop=True)

    else:
        y_hat = scores = None

    return([clf, scalings, ex_var, y_hat, scores])