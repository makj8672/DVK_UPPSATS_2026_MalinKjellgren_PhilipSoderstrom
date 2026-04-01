# Standalone rule-based trading strategy enhanced with logistic regression

# Psudo code:
# CLASS LogisticRegressionStrategy(RuleBasedStrategy):
#     FUNCTION train(dataframe):
#         Train logistic regression on indicator values (TODO: and target variable?)
#
#     FUNCTION generate_signal(row):
#         Get probability from logistic regression 
#         IF probability > 0.6:
#             RETURN 1 (buy signal)
#         IF probability < 0.4:
#             RETURN -1 (sell signal)
#         ELSE:
#             RETURN 0 (hold signal)

#from sklearn.linear_model import LogisticRegression TODO: swith to this?
from rule_based_strategy import RuleBasedStrategy

class LogisticRegressionStrategy(RuleBasedStrategy):
    def __init__(self, model):
        self.model = model
        pass

    