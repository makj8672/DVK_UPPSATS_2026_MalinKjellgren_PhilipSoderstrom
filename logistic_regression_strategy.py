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
    INDICATOR_COLUMNS = ["price_to_sma", "sma_cross", "rsi", "obv_diff"]
    REG_C = 10.0  # Default regularization strength, will be tuned on validation data

    # Constructor
    def __init__(self, model=None, scaler=None):
        self.model = model
        self.scaler = scaler

    def train(self, data_frame, C=None):
        """Train logistic regression model on training data and validate on validation data.
        
        Data is split chronologically:
        - 60% training
        - 20% validation (used for tuning)
        - 20% test (reserved for backtesting, not used here)
        """
        if C is not None:
            C = self.REG_C
        
        n = len(data_frame)
        train_end = int(n * 0.6)
        val_end = int(n * 0.8)

        X = data_frame[self.INDICATOR_COLUMNS]
        y = data_frame["target"]

        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]
        X_val = X.iloc[train_end:val_end]
        y_val = y.iloc[train_end:val_end]

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        self.model = LogisticRegression(
            class_weight="balanced",
            l1_ratio=1,
            solver="liblinear",
            random_state=42,
            C=C
        )
        self.model.fit(X_train_scaled, y_train)

        y_pred = self.model.predict(X_val_scaled)
        accuracy = accuracy_score(y_val, y_pred)
        print(f"Logistic Regression Validation Accuracy: {accuracy:.2f}")
        print("\n--- MQL5 Export ---")
        print(f"Intercept: {self.model.intercept_[0]}")
        print(f"Coefficients:")
        for feature, coef in zip(self.INDICATOR_COLUMNS, self.model.coef_[0]):
            print(f"  {feature}: {coef}")
        print(f"Scaler means: {self.scaler.mean_}")
        print(f"Scaler stds: {self.scaler.scale_}")

        proba = self.model.predict_proba(X_val_scaled)[:, 1]
        print(f"Min sannolikhet:   {proba.min():.3f}")
        print(f"Max sannolikhet:   {proba.max():.3f}")
        print(f"Medel sannolikhet: {proba.mean():.3f}")

    def generate_signal(self, row):
        """Used for live trading - returns buy/sell/hold signal based on probability."""
        latest = row[self.INDICATOR_COLUMNS].to_frame().T
        latest_scaled = self.scaler.transform(latest)
        probability = self.model.predict_proba(latest_scaled)[0][1]  # Probability of class 1 (buy signal)

        if probability > 0.50:
            return 1    # Buy signal
        elif probability < 0.49:
            return -1   # Sell signal
        else:
            return 0    # Hold signal

    def get_probability(self, row):
        """Return raw probability for interval-based backtesting."""
        latest = row[self.INDICATOR_COLUMNS].to_frame().T
        latest_scaled = self.scaler.transform(latest)
        return self.model.predict_proba(latest_scaled)[0][1]

    def tune(self, data_frame):
        """Tune C parameter on validatation data"""
        n = len(data_frame)
        train_end = int(n * 0.6)
        val_end = int(n * 0.8)

        x = data_frame[self.INDICATOR_COLUMNS]
        y = data_frame["target"]

        x_train = x.iloc[:train_end]
        y_train = y.iloc[:train_end]
        x_val = x.iloc[train_end:val_end]
        y_val = y.iloc[train_end:val_end]

        self.scaler = StandardScaler()
        x_train_scaled = self.scaler.fit_transform(x_train)
        x_val_scaled = self.scaler.transform(x_val)

        C_values = [0.01, 0.1, 1, 10, 100]
        best_C = None
        best_accuracy = 0

        print("\n--- Tuning C parameter ---")
        for C in C_values:
            model = LogisticRegression(
                class_weight="balanced",
                l1_ratio=1,
                solver="liblinear",
                random_state=42,
                C=C
            )
            model.fit(x_train_scaled, y_train)
            y_pred = model.predict(x_val_scaled)
            accuracy = accuracy_score(y_val, model.predict(x_val_scaled))
            print(f"C={C:<8} Accuracy = {accuracy:.3f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_C = C
            
        print(f"\nBästa C: {best_C} med accuracy: {best_accuracy:.3f}")
        return best_C


    #TODO: Not needed if we split data in train method, but could be useful for backtesting on unseen data
    def split_data(self, data_frame):
        # TODO: Implement data splitting logic
        pass