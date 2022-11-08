from decisionlist._base import (
    is_numeric,
    sort_rules,
    get_rules_from_forest,
    get_num_rules,
    num_rule_ind,
)
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.compose import make_column_transformer
import numpy as np


class DecisionListBase:
    """Generic class for decision lists"""

    def __init__(
        self,
        max_rule_length=3,
        min_confidence=0.85,
        min_support=0.025,
        num_trees=20,
        max_features_per_split=0.2,
        sign_digits=3,
    ):
        """Initializes the estimator"""
        self.max_rule_length = max_rule_length
        self.min_confidence = min_confidence
        self.min_support = min_support
        self.num_trees = num_trees
        self.max_features_per_split = max_features_per_split
        self.sign_digits = sign_digits
        self.is_fitted = False

        self._validate_params()
        self.is_fitted = False

    def get_params(self, deep=True):
        # suppose this estimator has parameters "alpha" and "recursive"
        return {
            "max_rule_length": self.max_rule_length,
            "min_confidence": self.min_confidence,
            "min_support": self.min_support,
            "num_trees": self.num_trees,
            "max_features_per_split": self.max_features_per_split,
            "sign_digits": self.sign_digits,
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def _validate_params(self):
        """Validates parameters upon initiation"""
        if self.max_rule_length > 3:
            raise ValueError(
                "rule length should not exceed 3 to ensure interpretability"
            )
        elif self.max_rule_length <= 0:
            raise ValueError("cannot interpret non positive rule length")
        else:
            self.max_rule_length = int(self.max_rule_length)

        if self.min_confidence > 1 or self.min_confidence <= 0:
            raise ValueError("minimum confidence should be in (0,1]")

        if self.min_support <= 0:
            raise ValueError("non positive support values cannot be interpreted")
        elif self.min_support >= 1:
            raise ValueError(
                "cannot interpet support values greater than or equal to 1"
            )

        if self.num_trees > 100 or self.num_trees < 5:
            raise ValueError("number of trees should be in [5, 100]")

        if self.max_features_per_split <= 0 or self.max_features_per_split > 1:
            raise ValueError(
                "maximum features to consider per split should be in (0,1]"
            )

    def _prefit_X(self, X_train):
        """Preprocesses X before train"""
        cat_cols = np.array(
            [i for i in range(X_train.shape[1]) if not is_numeric(X_train[:, i])]
        )

        if cat_cols.size > 0:
            self.oh_encoder = make_column_transformer(
                (OneHotEncoder(), cat_cols), remainder="passthrough"
            )
            self.oh_encoder.fit(X_train)
            X_train_t = self.oh_encoder.transform(X_train)
        else:
            X_train_t = X_train.copy()

        return X_train_t.astype(float)

    def _prefit_y(self, y_train):
        """Intended for classifier only"""
        if not is_numeric(y_train):
            self.label_transformer = LabelEncoder()
            y_train_t = self.label_transformer.fit_transform(y_train)
        else:
            y_train_t = y_train.copy()

        self.n_classes = np.unique(y_train).size

        return y_train_t.astype(float)

    def _prepredict_X(self, X_test):
        """Preprocesses X before test"""
        cat_cols = np.array(
            [i for i in range(X_test.shape[1]) if not is_numeric(X_test[:, i])]
        )

        if cat_cols.size > 0:
            X_test_t = self.oh_encoder.transform(X_test)
        else:
            X_test_t = X_test.copy()

        return X_test_t.astype(float)

    def _prepredict_y(self, y_test):
        """Intended for classifier only"""
        if self.label_transformer is not None:
            y_test_t = self.label_transformer.transform(y_test)
        else:
            y_test_t = y_test.copy()

        return y_test_t.astype(float)

    def _check_if_fitted(self):
        """Checks whether is fitted"""
        if not self.is_fitted:
            raise Exception("The estimator has not been fitted")


class DecisionListClassifier(DecisionListBase):
    """Fits a Decision List Classifier implementing sequential covering principles"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.label_transformer = None

    def fit(self, X_train, y_train):
        """Fit the classifier"""

        X_train_t = self._prefit_X(X_train)

        y_train_t = self._prefit_y(y_train)

        self._min_support_n = max(round(1.0 * self.min_support * X_train.shape[0]), 1)

        self.rules = []

        i_ = 0

        X_train_remaining, y_train_remaining = X_train_t, y_train_t
        # sequential covering
        while True:

            # train forest
            rf = RandomForestClassifier(
                max_depth=self.max_rule_length,
                n_estimators=self.num_trees,
                max_features=self.max_features_per_split,
                bootstrap=False,
            )

            rf.fit(X_train_remaining, y_train_remaining)

            # extract rules from the forest
            forest_rules = get_rules_from_forest(
                rf,
                min_confidence=self.min_confidence,
                min_support=self._min_support_n,
                concise=True,
            )

            if len(forest_rules) == 0:
                # if no rule was found
                if i_ == 0:
                    raise Exception(
                        f"No rules found with confidence > {self.min_confidence} and support > {self.min_support}"
                    )
                else:
                    c, v = np.unique(y_train_remaining, return_counts=1)

                    predicted_class = c[v.argmax()]
                    confidence = round(v[v.argmax()] / v.sum(), self.sign_digits)
                    support = X_train_remaining.shape[0]
                    self.rules += [
                        (("__else",), predicted_class, v.tolist(), confidence, support)
                    ]

                    break

            else:
                sorted_rules = sort_rules(forest_rules)
                best_rule = sorted_rules[0]
                num_rules = get_num_rules(best_rule)

                rule_ind = num_rule_ind(X_train_remaining, num_rules)

                self.rules += [best_rule]

                X_train_remaining, y_train_remaining = (
                    X_train_remaining[~rule_ind],
                    y_train_remaining[~rule_ind],
                )

            if X_train_remaining.shape[0] == 0:
                # if no more data stop
                break

            rf = None

            i_ += 1

        self.is_fitted = True

    def predict(self, X_test):
        """Predicts labels for new samples"""

        self._check_if_fitted()
        X_test_t = self._prepredict_X(X_test)

        y_pred = np.repeat(np.nan, X_test_t.shape[0])
        already_classified = np.repeat(False, X_test_t.shape[0])

        for i in range(len(self.rules)):
            rule = self.rules[i]
            if rule[0][0] != "__else":
                num_rules = get_num_rules(rule)
                rule_ind = num_rule_ind(X_test_t, num_rules)
                y_pred[(~already_classified) & (rule_ind)] = rule[1]
                already_classified = np.where(
                    (already_classified) | (rule_ind), True, False
                )
            else:
                y_pred[(~already_classified)] = rule[1]
        return y_pred

    def predict_proba(self, X_test):
        """Predicts probabilities for new samples"""
        self._check_if_fitted()
        X_test_t = self._prepredict_X(X_test)

        y_pred = np.repeat(np.nan, X_test_t.shape[0] * self.n_classes).reshape(
            X_test_t.shape[0], self.n_classes
        )
        already_classified = np.repeat(False, X_test_t.shape[0])

        for i in range(len(self.rules)):
            rule = self.rules[i]
            if rule[0][0] != "__else":
                num_rules = get_num_rules(rule)
                rule_ind = num_rule_ind(X_test_t, num_rules)
                y_pred[(~already_classified) & (rule_ind)] = np.array(rule[2]) / sum(
                    rule[2]
                )
                already_classified = np.where(
                    (already_classified) | (rule_ind), True, False
                )
            else:
                y_pred[(~already_classified)] = np.array(rule[2]) / sum(rule[2])
        return y_pred

    def score(self, X_test, y_test):
        """Returns the accuracy for new X, y data"""
        y_pred = self.predict(X_test)
        y_test_t = self._prepredict_y(y_test)
        return (y_pred == y_test_t).sum() / y_test.size


class DecisionListRegressor(DecisionListBase):
    """Fits a Decision List Classifier implementing sequential covering principles"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit(self, X, y):
        """Fit the classifier"""
        pass

    def predict(self, X_test):
        """Predicts labels for new samples"""
        pass

    def score(self, X_test, y_test):
        """Returns the accuracy for new X, y data"""
        pass
