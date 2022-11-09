import numpy as np
from sklearn.tree import _tree
from sklearn.tree._classes import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier


def is_numeric(vector: np.ndarray):
    """Checks if a vector is numeric"""
    vector = np.unique(vector[vector != None])
    try:
        vector.astype(np.float64)
        return True
    except:
        return False


def mine_tree_rules(tree, feature_names=None, class_names=None, sign_digits=3):
    """copied and modified from:
    https://mljar.com/blog/extract-rules-decision-tree/
    """
    tree_ = tree.tree_

    if feature_names == None:
        feature_names = [f"__col{i}" for i in range(tree.n_features_in_)]

    if class_names is None and isinstance(tree, DecisionTreeClassifier):
        class_names = tree.classes_

    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]

    paths = []
    path = []

    def recurse(node, path, paths):

        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            p1, p2 = list(path), list(path)
            p1 += [f"({name} <= {np.round(threshold, sign_digits)})"]
            recurse(tree_.children_left[node], p1, paths)
            p2 += [f"({name} > {np.round(threshold, sign_digits)})"]
            recurse(tree_.children_right[node], p2, paths)
        else:

            if isinstance(tree, DecisionTreeRegressor):
                path += [
                    (
                        tree_.value[node],
                        tree_.impurity[node],
                        tree_.n_node_samples[node],
                    )
                ]

            else:
                path += [(tree_.value[node], tree_.n_node_samples[node])]
            paths += [path]

    recurse(0, path, paths)

    # sort by samples count
    samples_count = [p[-1][1] for p in paths]
    ii = list(np.argsort(samples_count))
    paths = [paths[i] for i in reversed(ii)]

    rules = []
    for path in paths:
        rule_list = []
        rule = []
        for p in path[:-1]:
            rule += [str(p)]
        rule_list += [tuple(rule)]

        if isinstance(tree, DecisionTreeRegressor):
            # regression

            rule_list += [np.round(path[-1][0][0][0], 3)]  # value
            rule_list += [
                0 if path[-1][-1] <= 1 else round(path[-1][-2] ** (0.5), 3)
            ]  # rmse
        else:
            # classification

            class_counts = path[-1][0][0]
            l = np.argmax(class_counts)

            rule_list += [class_names[l]]
            rule_list += [class_counts.tolist()]
            rule_list += [
                np.round(1.0 * class_counts[l] / np.sum(class_counts), sign_digits)
            ]  # confidence
        rule_list += [path[-1][-1]]  # support
        rules += [tuple(rule_list)]

    return rules


def make_rules_concise(rules, one_hot_encoder):

    """Makes rules more concise"""
    rules_c = []

    # if a numerical feature is used twice with the same inequality operator merge
    for i in range(len(rules)):
        rule = rules[i]

        # if rule is empty skip
        if len(rule[0]) == 0:
            continue

        features = [
            r.split("<=")[0].replace("(", "").strip()
            if "<" in r
            else r.split(">")[0].replace("(", "").strip()
            for r in rule[0]
        ]
        inequalities = ["<=" if "<" in r else ">" for r in rule[0]]
        values = [
            r.split("<=")[1].replace(")", "").strip()
            if "<" in r
            else r.split(">")[1].replace(")", "").strip()
            for r in rule[0]
        ]
        conditions_array = np.array([features, inequalities, values]).T

        unique_features = np.unique(features, return_counts=True)

        # if there are features that exist more than once in the rule conditions
        if unique_features[1].max() > 1:
            redundant_indices = []
            for feature in unique_features[0][unique_features[1] > 1]:
                conditions_filtered = conditions_array[
                    conditions_array[:, 0] == feature
                ]
                unique_ineq = np.unique(conditions_filtered[:, 1], return_counts=True)

                # if there are more than one occurences for an inequality
                if unique_ineq[1].max() > 1:
                    for ineq in unique_ineq[0][unique_ineq[1] > 1]:
                        if ineq == ">":

                            # keep only the maximum
                            max_val = (
                                conditions_filtered[
                                    conditions_filtered[:, 1] == ineq, 2
                                ]
                                .astype(float)
                                .max()
                            )

                            redundant_indices += np.where(
                                (conditions_array[:, 0] == feature)
                                & (conditions_array[:, 1] == ineq)
                                & (conditions_array[:, 2].astype(float) < max_val)
                            )[0].tolist()

                        else:
                            # keep only the minimum
                            min_val = (
                                conditions_filtered[
                                    conditions_filtered[:, 1] == ineq, 2
                                ]
                                .astype(float)
                                .min()
                            )

                            redundant_indices += np.where(
                                (conditions_array[:, 0] == feature)
                                & (conditions_array[:, 1] == ineq)
                                & (conditions_array[:, 2].astype(float) > min_val)
                            )[0].tolist()

            rules_c += [
                tuple(
                    [
                        tuple(
                            rules[i][0][j]
                            for j in range(len(rules[i][0]))
                            if j not in redundant_indices
                        ),
                        rules[i][1],
                        rules[i][2],
                        rules[i][3],
                        rules[i][4],
                    ]
                    if len(rules[i]) == 5
                    else [
                        tuple(
                            rules[i][0][j]
                            for j in range(len(rules[i][0]))
                            if j not in redundant_indices
                        ),
                        rules[i][1],
                        rules[i][2],
                        rules[i][3],
                    ]
                )
            ]

        else:
            rules_c += [rule]

    # return unique
    return [rules_c[i] for i in range(len(rules_c)) if rules_c[i] not in rules_c[:i]]

    # if a numerical feature is used twice with the same inequality operator merge
    for i in range(len(rules)):
        rule = rules[i]

        # if rule is empty skip
        if len(rule[0]) == 0:
            continue

        features = [
            r.split("<=")[0].replace("(", "").strip()
            if "<" in r
            else r.split(">")[0].replace("(", "").strip()
            for r in rule[0]
        ]
        inequalities = ["<=" if "<" in r else ">" for r in rule[0]]
        values = [
            r.split("<=")[1].replace(")", "").strip()
            if "<" in r
            else r.split(">")[1].replace(")", "").strip()
            for r in rule[0]
        ]
        conditions_array = np.array([features, inequalities, values]).T

        unique_features = np.unique(features, return_counts=True)

        # if there are features that exist more than once in the rule conditions
        if unique_features[1].max() > 1:
            redundant_indices = []
            for feature in unique_features[0][unique_features[1] > 1]:
                conditions_filtered = conditions_array[
                    conditions_array[:, 0] == feature
                ]
                unique_ineq = np.unique(conditions_filtered[:, 1], return_counts=True)

                # if there are more than one occurences for an inequality
                if unique_ineq[1].max() > 1:
                    for ineq in unique_ineq[0][unique_ineq[1] > 1]:
                        if ineq == ">":

                            # keep only the maximum
                            max_val = (
                                conditions_filtered[
                                    conditions_filtered[:, 1] == ineq, 2
                                ]
                                .astype(float)
                                .max()
                            )

                            redundant_indices += np.where(
                                (conditions_array[:, 0] == feature)
                                & (conditions_array[:, 1] == ineq)
                                & (conditions_array[:, 2].astype(float) < max_val)
                            )[0].tolist()

                        else:
                            # keep only the minimum
                            min_val = (
                                conditions_filtered[
                                    conditions_filtered[:, 1] == ineq, 2
                                ]
                                .astype(float)
                                .min()
                            )

                            redundant_indices += np.where(
                                (conditions_array[:, 0] == feature)
                                & (conditions_array[:, 1] == ineq)
                                & (conditions_array[:, 2].astype(float) > min_val)
                            )[0].tolist()

            rules_c += [
                tuple(
                    [
                        tuple(
                            rules[i][0][j]
                            for j in range(len(rules[i][0]))
                            if j not in redundant_indices
                        ),
                        rules[i][1],
                        rules[i][2],
                        rules[i][3],
                        rules[i][4],
                    ]
                )
            ]

        else:
            rules_c += [rule]

    # return unique
    return [rules_c[i] for i in range(len(rules_c)) if rules_c[i] not in rules_c[:i]]


def get_rules_from_forest(
    forest, min_confidence, min_support, concise=True, sign_digits=3
):
    """extracts rules from a random forest"""
    all_rules = []

    for dt in forest.estimators_:
        if concise:
            all_rules += make_rules_concise(
                mine_tree_rules(dt, sign_digits=sign_digits), None
            )

        else:
            all_rules += mine_tree_rules(dt)

        if isinstance(forest, RandomForestClassifier):

            return [
                r
                for r in all_rules
                if (r[-2] >= min_confidence) and (r[-1] >= min_support)
            ]
        else:
            return [r for r in all_rules if (r[-1] >= min_support)]


def sort_rules(rules, mode="classification"):
    """Sorts rules by confidence then support"""

    rule_array = np.array(
        [[i, rules[i][-2], rules[i][-1], len(rules[i][0])] for i in range(len(rules))]
    )

    if mode == "classification":
        # classification

        # sort by confidence, support and rule length
        sorted_idx = np.lexsort(
            (
                rule_array[:, 3],
                -rule_array[:, 2],
                -rule_array[:, 1],
            )
        ).astype(int)
    else:

        # sort by rmse, support and rule length
        sorted_idx = np.lexsort(
            (
                rule_array[:, 3],
                -rule_array[:, 2],
                rule_array[:, 1],
            )
        ).astype(int)

    return [rules[i] for i in sorted_idx]


def get_num_rules(rule):
    """Transform text rule to tuple,
    1st element being col index, second binary (0 left), third threshold"""

    n_rule = []
    for r in rule[0]:
        if "<=" in r:
            spl = r.split("<=")
            n_rule += [
                (
                    int(spl[0].replace("(__col", "").strip()),
                    0,
                    float(spl[1].replace(")", "").strip()),
                )
            ]
        else:
            spl = r.split(">")
            n_rule += [
                (
                    int(spl[0].replace("(__col", "").strip()),
                    1,
                    float(spl[1].replace(")", "").strip()),
                )
            ]

    return n_rule


def num_rule_ind(X, rule):
    """Filters data using num rule"""
    rule_ind = None

    for num_rule in rule:
        if num_rule[1] == 0:
            # if rule is less than or equal

            if rule_ind is None:
                # if it is the first rule

                rule_ind = np.where(
                    X[:, num_rule[0]] <= num_rule[2],
                    True,
                    False,
                )
            else:
                # if there are more than 1 rules
                rule_ind = (rule_ind) & (
                    np.where(
                        X[:, num_rule[0]] <= num_rule[2],
                        True,
                        False,
                    )
                )
        else:
            # if rule is greater than

            if rule_ind is None:
                # if it is the first rule
                rule_ind = np.where(X[:, num_rule[0]] > num_rule[2], True, False)
            else:
                # if there are more than 1 rules
                rule_ind = (rule_ind) & (
                    np.where(
                        X[:, num_rule[0]] > num_rule[2],
                        True,
                        False,
                    )
                )

    return rule_ind
