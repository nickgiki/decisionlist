from decisionlist.estimators import DecisionListClassifier
from decisionlist._base import (
    is_numeric,
    sort_rules,
    get_rules_from_forest,
    get_num_rules,
    num_rule_ind,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification, load_iris
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

results = []


for i in range(50):

    # create data
    X, y = make_classification(1000, n_features=10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # fit the dl classifier
    dlc = DecisionListClassifier(min_support=0.02, min_confidence=0.75)
    dlc.fit(X_train, y_train)

    # fit the regressor
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)

    results += [dlc.score(X_test, y_test) / dtc.score(X_test, y_test) - 1]

plt.hist(results)
plt.title("DecisionList improvement over DecisionTree")
np.mean(results)
