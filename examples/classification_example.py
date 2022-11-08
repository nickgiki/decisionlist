from decisionlist._base import np, get_rules_from_forest, is_numeric, sort_rules
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


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

X, y = data[:, :-1], data[:, -1]

cat_cols = np.array([i for i in range(X.shape[1]) if not is_numeric(X[:, i])])
num_cols = np.array([i for i in range(X.shape[1]) if i not in cat_cols])

X = np.concatenate((X[:, cat_cols], X[:, num_cols]), axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=3
)


oh = OneHotEncoder()
rf = RandomForestClassifier(max_depth=2, n_estimators=40)
re = LabelEncoder()

transformer = make_column_transformer(
    (OneHotEncoder(), cat_cols), remainder="passthrough"
)

if not is_numeric(y):
    y_train_t = re.fit_transform(y_train)
    y_test_t = re.fit_transform(y_test)
else:
    y_train_t, y_test_t = y_train, y_test

X_train_t = transformer.fit_transform(X_train).astype(float)
X_test_t = transformer.transform(X_test).astype(float)


rf.fit(X_train_t, y_train_t)

sort_rules(get_rules_from_forest(rf, 0.8, 6))[0]
