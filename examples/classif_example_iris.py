from decisionlist.estimators import DecisionListClassifier
from decisionlist._base import (
    is_numeric,
    sort_rules,
    get_rules_from_forest,
    get_num_rules,
    num_rule_ind,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

results = []


# create data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# fit the dl classifier
dlc = DecisionListClassifier(min_support=0.02, min_confidence=0.75)
dlc.fit(X_train, y_train)

# fit the regressor
dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)


np.mean(round(dlc.score(X_test, y_test) / dtc.score(X_test, y_test) - 1, 3))
