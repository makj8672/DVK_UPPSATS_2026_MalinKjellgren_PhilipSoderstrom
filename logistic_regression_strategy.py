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
from sklearn.metrics import accuracy_score
from data_pipeline import split_data
#import joblib TODO: Delete or save?

from rule_based_strategy import RuleBasedStrategy

class LogisticRegressionStrategy(RuleBasedStrategy):
    INDICATOR_COLUMNS = ["price_to_sma", "sma_cross", "rsi", "obv_diff"]
    REG_C = 10.0  # Default regularization strength, will be tuned on validation data

    # Constructor
    def __init__(self, model=None, scaler=None):
        self.model = model
        self.scaler = scaler

    def _prepare_data(self, train_data, val_data):
        """Prepare and scale features for training and validation.
        
        Returns scaled train and validation sets.
        """
        
        x_train = train_data[self.INDICATOR_COLUMNS]
        y_train = train_data["target"]
        x_val = val_data[self.INDICATOR_COLUMNS]
        y_val = val_data["target"]

        self.scaler = StandardScaler()
        x_train_scaled = self.scaler.fit_transform(x_train)
        x_val_scaled = self.scaler.transform(x_val)

        return x_train_scaled, y_train, x_val_scaled, y_val
    
    def tune(self, train_data, val_data):
        """Tune C parameter on validatation data"""
        
        x_train_scaled, y_train, x_val_scaled, y_val = self._prepare_data(train_data, val_data)

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
            accuracy = accuracy_score(y_val, model.predict(x_val_scaled))
            print(f"C={C:<8} Accuracy = {accuracy:.3f}")
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_C = C
            
        print(f"\nBästa C: {best_C} med accuracy: {best_accuracy:.3f}")
        return best_C
    
    def train(self, train_data, val_data, C=None):
        """Train logistic regression model on training data and validate on validation data.
        
        Data preparation and scaling is handled by _prepare_data().
        If C is not provided, REG_C is used as default regularization strength.
        Prints validation accuracy, MQL5 export values, and probability distribution.
        """
        if C is None:
            C = self.REG_C

        x_train_scaled, y_train, x_val_scaled, y_val = self._prepare_data(train_data, val_data)

        self.model = LogisticRegression(
            class_weight="balanced",
            l1_ratio=1,
            solver="liblinear",
            random_state=42,
            C=C
        )
        self.model.fit(x_train_scaled, y_train)

        y_pred = self.model.predict(x_val_scaled)
        accuracy = accuracy_score(y_val, y_pred)
        print(f"Logistic Regression Validation Accuracy: {accuracy:.2f}")
        print("\n--- MQL5 Export ---")
        print(f"Intercept: {self.model.intercept_[0]}")
        print(f"Coefficients:")
        for feature, coef in zip(self.INDICATOR_COLUMNS, self.model.coef_[0]):
            print(f"  {feature}: {coef}")
        print(f"Scaler means: {self.scaler.mean_}")
        print(f"Scaler stds: {self.scaler.scale_}")

        proba = self.model.predict_proba(x_val_scaled)[:, 1]
        print(f"Min sannolikhet:   {proba.min():.3f}")
        print(f"Max sannolikhet:   {proba.max():.3f}")
        print(f"Medel sannolikhet: {proba.mean():.3f}")
   
    def get_probability(self, row):
        """Return raw probability for interval-based backtesting."""
        latest = row[self.INDICATOR_COLUMNS].to_frame().T
        latest_scaled = self.scaler.transform(latest)
        return self.model.predict_proba(latest_scaled)[0][1]

    def generate_signal(self, row):
            """
            NOTE: Not used in current implementation.
            Live trading is handled by the MQL5 Expert Advisor.
            Backtesting uses get_probability() instead.
            Could be used if live trading is moved to Python in the future.
            """
            latest = row[self.INDICATOR_COLUMNS].to_frame().T
            latest_scaled = self.scaler.transform(latest)
            probability = self.model.predict_proba(latest_scaled)[0][1]  # Probability of class 1 (buy signal)

            if probability > 0.50:
                return 1    # Buy signal
            elif probability < 0.49:
                return -1   # Sell signal
            else:
                return 0    # Hold signal
