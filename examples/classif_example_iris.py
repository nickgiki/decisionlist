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

n_estimators = []

for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    
    # fit the dl classifier
    dlc = DecisionListClassifier(min_support=0.01,max_features_per_split=0.85)
    dlc.fit(X_train, y_train)
    
    # fit the regressor
    dtc = DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    
    results += [np.mean(round(dlc.score(X_test, y_test) / dtc.score(X_test, y_test) - 1, 3))]

np.quantile(results,q=[0.025,0.5,0.975])
