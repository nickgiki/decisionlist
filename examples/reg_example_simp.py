from decisionlist.estimators import DecisionListRegressor
from decisionlist._base import (
    mine_tree_rules,
    is_numeric,
    sort_rules,
    get_rules_from_forest,
    get_num_rules,
    num_rule_ind,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

results = []


np.random.seed(1)

for i in range(50):

    X, y = make_regression(1000, n_features=10)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # fit the dl classifier
    dlr = DecisionListRegressor()
    dlr.fit(X_train, y_train)

    # fit the regressor
    dtr = DecisionTreeRegressor(max_depth=2)
    dtr.fit(X_train, y_train)

    results += [dlr.score(X_test, y_test) / dtr.score(X_test, y_test) - 1]

plt.hist(results)
