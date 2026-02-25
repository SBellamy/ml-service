import os

import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def train_model(df: pd.DataFrame) -> tuple[BaseEstimator, pd.DataFrame, pd.Series]:
    # separate input features from the label column
    X = df.drop(columns=["target"])
    y = df["target"]

    # split data into training and test subsets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # build a simple sklearn pipeline
    model: BaseEstimator = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                max_iter=1000, 
                random_state=42, 
                class_weight="balanced" if os.getenv("BALANCED_CLASS_WEIGHT") == "True" else None
                )
            ),
        ]
    )

    # fit pipeline on training data
    model.fit(X_train, y_train)

    # return trained model and test data sets (for later evaluation)
    return model, X_test, y_test
