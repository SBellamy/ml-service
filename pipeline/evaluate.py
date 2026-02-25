from sklearn.base import BaseEstimator
import pandas as pd
from sklearn.metrics import f1_score

def evaluate(model: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    # use the trained model to predict labels for the test set
    y_pred = model.predict(X_test)  # type: ignore

    # compute the F1 score between true labels and predicted labels
    score = f1_score(y_test, y_pred)

    # type-cast (for json serialization) and return the F1 score
    return float(score)