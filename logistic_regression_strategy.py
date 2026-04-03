#logistic_regression_strategy.py
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

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#import joblib TODO: Delete or save?

from rule_based_strategy import RuleBasedStrategy

class LogisticRegressionStrategy(RuleBasedStrategy):
    INDICATOR_COLUMNS = ["price_to_sma200", "sma_cross", "rsi", "obv_diff"]

    # Constructor
    def __init__(self, model=None, scaler=None):
        self.model = model
        self.scaler = scaler

    def train(self, data_frame):
        X = data_frame[self.INDICATOR_COLUMNS]
        y = data_frame["target"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
            )
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        self.model = LogisticRegression(class_weight="balanced")
        self.model.fit(X_train_scaled, y_train)

        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Logistic Regression Accuracy: {accuracy:.2f}")

        # Debug
        proba = self.model.predict_proba(X_test_scaled)[:, 1]
        print(f"Min sannolikhet:   {proba.min():.3f}")
        print(f"Max sannolikhet:   {proba.max():.3f}")
        print(f"Medel sannolikhet: {proba.mean():.3f}")

    def generate_signal(self, row):
        latest = row[self.INDICATOR_COLUMNS].to_frame().T
        latest_scaled = self.scaler.transform(latest)
        probability = self.model.predict_proba(latest_scaled)[0][1]  # Probability of class 1 (buy signal)

        if probability > 0.50:
            return 1    # Buy signal
        elif probability < 0.49:
            return -1   # Sell signal
        else:
            return 0    # Hold signal


    #TODO: Not needed if we split data in train method, but could be useful for backtesting on unseen data
    def split_data(self, data_frame):
        # TODO: Implement data splitting logic
        pass