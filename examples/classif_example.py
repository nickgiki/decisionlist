from decisionlist.estimators import DecisionListClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def score_dlc(X_train, X_test, y_train, y_test):

    dlc = DecisionListClassifier()

    # fit the classifier
    dlc.fit(X_train, y_train)

    # score
    return dlc.score(X_test, y_test)


def score_dtc(X_train, X_test, y_train, y_test):
    dtc = DecisionTreeClassifier(max_depth=3)

    # fit the classifier
    dtc.fit(X_train, y_train)

    # score
    return dtc.score(X_test, y_test)


dlc_score = []
dtc_score = []

for i in range(50):
    X, y = make_classification(1000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    dlc_score.append(score_dlc(X_train, X_test, y_train, y_test))
    dtc_score.append(score_dtc(X_train, X_test, y_train, y_test))

print(sum([dlc_score[i] >= dtc_score[i] for i in range(len(dlc_score))]))

plt.hist([dlc_score[i] / dtc_score[i] - 1 for i in range(len(dlc_score))])
