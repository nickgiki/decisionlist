from decisionlist._base import make_rules_concise

redundant_rules = [
    (("(c1 > 4)", "(c1 > 3)", "(c2 <= 3)", "(c2 <= 1)"),1,[1, 9], 0.9, 10),
    (("(c3 > 8)", "(c3 > 6)", "(c3 > 2)", "(c1 <= 2)"),0,[3, 2], 0.8, 5),
]

print(make_rules_concise(redundant_rules, None))
