from decisionlist._base import np, mine_tree_rules, is_numeric
from decisionlist.estimators import DecisionListClassifier


data = np.array(
    [
        ["B", 8.2, "No"],
        ["B", 8.8, "Yes"],
        ["B", 11.0, "No"],
        ["B", 10.0, "Yes"],
        ["B", 8.1, "Yes"],
        ["B", 7.1, "No"],
        ["B", 7.9, "No"],
        ["C", -0.2, "Yes"],
        ["B", 7.9, "No"],
        ["A", 9.1, "No"],
        ["A", 9.2, "No"],
        ["C", 2.2, "No"],
        ["B", 5.6, "No"],
        ["B", 7.3, "No"],
        ["B", 7.9, "Yes"],
        ["B", 7.0, "No"],
        ["A", 10.6, "Yes"],
        ["A", 10.2, "No"],
        ["C", 2.5, "Yes"],
        ["C", -0.8, "Yes"],
    ]
)

colnames = ["categorical_1", "num", "binary"]

dlc = DecisionListClassifier(min_support=0.08)

X_train, y_train = data[:, :-1], data[:, -1]

dlc.fit(X_train, y_train)
