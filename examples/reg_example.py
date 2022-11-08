from decisionlist.estimators import DecisionListRegressor
from decisionlist._base import mine_tree_rules, make_rules_concise, sort_rules
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def score_dlc(X_train, X_test, y_train, y_test):

    dlc = DecisionListRegressor()

    # fit the classifier
    dlc.fit(X_train, y_train)

    # score
    return dlc.score(X_test, y_test)


def score_dtc(X_train, X_test, y_train, y_test):
    dtr = DecisionTreeRegressor(max_depth=2)

    # fit the classifier
    dtr.fit(X_train, y_train)

    # score
    return dtr.score(X_test, y_test)


dlc_score = []
dtc_score = []

for i in range(1):
    X, y = make_regression(1000, n_features=10)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=3
    )

    dlc_score.append(score_dlc(X_train, X_test, y_train, y_test))
    dtc_score.append(score_dtc(X_train, X_test, y_train, y_test))

print(sum([dlc_score[i] >= dtc_score[i] for i in range(len(dlc_score))]))

plt.hist([dlc_score[i] / dtc_score[i] - 1 for i in range(len(dlc_score))])
