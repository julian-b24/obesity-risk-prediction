import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV

import warnings
warnings.filterwarnings('ignore')

N_JOBS = 7


def build_models(x, y):
    logreg = __train_lr(x, y)
    rf = __train_rf(x, y)
    gb = __train_gboost(x, y)
    return logreg, rf, gb


def build_lr(x, y, penalty: str | None, solver: str):
    logreg = LogisticRegression(
        max_iter=300,
        penalty=penalty,
        solver=solver
        )
    
    logreg.fit(x, y)
    return logreg


def build_xgboost(x, y, learning_rate: float, n_estimators: int):
    gboost = GradientBoostingClassifier(
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        n_iter_no_change=20
        )
    
    gboost.fit(x, y)
    return gboost


def build_rf(x, y, n_estimators: int):
    rf = RandomForestClassifier(
        n_estimators=n_estimators
        )
    
    rf.fit(x, y)
    return rf


def __train_lr(x, y):
    logreg = LogisticRegression(max_iter=300)
    parameters = {
        "penalty": ['l1', 'l2', None],
        'solver': ['saga', 'lbfgs', 'newton-c'],
    }

    print('Started Grid Search - Model: Logistic Regression Classifier')
    clf = GridSearchCV(logreg, param_grid=parameters, n_jobs=N_JOBS, verbose=4)
    clf.fit(x, y)
    print("Best parameters Logistic Regression Classifier: {}".format(clf.best_params_))
    print('Finished Grid Search - Model: Logistic Regression Classifier')
    return clf


def __train_gboost(x, y):
    gboost = GradientBoostingClassifier()
    parameters = {
        'learning_rate': [10, 1, 0.1, 0.01, 0.001, 0.0001],
        'n_estimators': [24, 48, 64, 100],
        'n_iter_no_change': [20]
    }
    print('Started Grid Search - Model: Gradient Boosting Classifier')
    clf = GridSearchCV(gboost, param_grid=parameters, n_jobs=N_JOBS, verbose=4)
    clf.fit(x, y)
    print("Best parameters Gradient Boosting Classifier: {}".format(clf.best_params_))
    print('Finished Grid Search - Model: Gradient Boosting Classifier')
    return clf


def __train_rf(x, y):
    rf = RandomForestClassifier()
    parameters = {
        'n_estimators': [20, 40, 60, 80, 100, 150, 200, 300]
    }
    print('Started Grid Search - Model: Random Forest Classifier')
    clf = GridSearchCV(rf, param_grid=parameters, n_jobs=N_JOBS, verbose=4)
    clf.fit(x, y)
    print("Best parameters Random Forest Classifier: {}".format(clf.best_params_))
    print('Finished Grid Search - Model: Random Forest')
    return clf