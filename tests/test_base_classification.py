
from decisionlist._base import (
    is_numeric,
    sort_rules,
    get_rules_from_forest,
    get_num_rules,
    sort_rules,
    num_rule_ind,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


X,y = make_classification()
X_tr, X_ts, y_tr, y_ts = train_test_split(X,y)


rfc = RandomForestClassifier(max_depth=3,n_estimators=3)
rfc.fit(X_tr,y_tr)

min_support,min_confidence = 5,0.7
forest_rules = get_rules_from_forest(rfc,min_confidence,min_support)
assert min([r[-1] for r in forest_rules]) >= min_support, "min_support is greater than rule support"
assert min([r[-2] for r in forest_rules]) >= min_confidence, "min_confidence is greater than rule confidence"

sorted_rules = sort_rules(forest_rules)

for i in range(1,len(sorted_rules)):
    assert sorted_rules[i][-2]<= sorted_rules[i-1][-2], f"confidence is not sorted {i}, {sorted_rules[i-1][-2]}, {sorted_rules[i][-2]}"
    if sorted_rules[i][-2] == sorted_rules[i-1][-2]:
        assert sorted_rules[i][-1]<= sorted_rules[i-1][-1], "support is not sorted"
        if sorted_rules[i][-1] == sorted_rules[i-1][-1]:
            assert len(sorted_rules[i][0]) == len(sorted_rules[i-1][0]), "rule length is not sorted"