#logistic_regression_strategy.py
# Rule-based strategy enhanced with logistic regression (filters / scores rule direction)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

from rule_based_strategy import RuleBasedStrategy


class LogisticRegressionStrategy(RuleBasedStrategy):
    """LR on indicator rows where the rule fires; signal_sign tells long vs short."""

    INDICATOR_COLUMNS = ["price_to_sma", "sma_cross", "rsi", "obv_diff"]
    FEATURE_COLUMNS = INDICATOR_COLUMNS + ["signal_sign"]
    REG_C = 4.64
    CONFIRM_THRESHOLD = 0.50 #TODO: Var 0.52 innan

    def __init__(self, model=None, scaler=None):
        self.model = model
        self.scaler = scaler

    def _prepare_data(self, train_data, val_data):
        x_train = train_data[self.FEATURE_COLUMNS]
        y_train = train_data["target"]
        x_val = val_data[self.FEATURE_COLUMNS]
        y_val = val_data["target"]

        self.scaler = StandardScaler()
        x_train_scaled = self.scaler.fit_transform(x_train)
        x_val_scaled = self.scaler.transform(x_val)

        return x_train_scaled, y_train, x_val_scaled, y_val

    def tune(self, train_data, val_data):
        """Tune C on validation accuracy (same setup as train)."""
        x_train_scaled, y_train, x_val_scaled, y_val = self._prepare_data(train_data, val_data)

        C_values = np.logspace(-2, 2, 10).tolist()
        best_C = None
        best_accuracy = 0

        print("\n--- Tuning C parameter ---")
        for C in C_values:
            model = LogisticRegression(
                class_weight="balanced",
                l1_ratio=1, #TODO penalty="l2" tidigare
                solver="liblinear",
                random_state=42,
                C=C,
            )
            model.fit(x_train_scaled, y_train)
            accuracy = accuracy_score(y_val, model.predict(x_val_scaled))
            print(f"C={C:<10.4f} Accuracy = {accuracy:.3f}")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_C = C

        print(f"\nBästa C: {best_C} med accuracy: {best_accuracy:.3f}")
        return best_C

    def train(self, train_data, val_data, C=None):
        if C is None:
            C = self.REG_C

        x_train_scaled, y_train, x_val_scaled, y_val = self._prepare_data(train_data, val_data)

        self.model = LogisticRegression(
            class_weight="balanced",
            penalty="l2",
            solver="liblinear",
            random_state=42,
            C=C,
        )
        self.model.fit(x_train_scaled, y_train)

        y_pred = self.model.predict(x_val_scaled)
        accuracy = accuracy_score(y_val, y_pred)
        print(f"Logistic Regression Validation Accuracy: {accuracy:.2f}")
        print("\n--- MQL5 Export ---")
        print(f"Intercept: {self.model.intercept_[0]}")
        print("Coefficients:")
        for feature, coef in zip(self.FEATURE_COLUMNS, self.model.coef_[0]):
            print(f"  {feature}: {coef}")
        print(f"Scaler means: {self.scaler.mean_}")
        print(f"Scaler stds: {self.scaler.scale_}")

        proba = self.model.predict_proba(x_val_scaled)[:, 1]
        print(f"Min sannolikhet:   {proba.min():.3f}")
        print(f"Max sannolikhet:   {proba.max():.3f}")
        print(f"Medel sannolikhet: {proba.mean():.3f}")

    def get_probability(self, row, signal):
        """P(success): favorable next bar for the given rule direction (1=long, -1=short)."""
        sign = 1.0 if signal == 1 else -1.0                         # Tell LR which direction we're asking about (long vs short)
        
        # DataFrame with same column names/order as fit — avoids sklearn feature-name warnings
        data = {c: float(row[c]) for c in self.INDICATOR_COLUMNS}   # Extract indicator values from row
        data["signal_sign"] = sign                                  # Add signal_sign to the input data
        X = pd.DataFrame([data], columns=self.FEATURE_COLUMNS)      # Create DataFrame with correct column order for scaler/model
        X_scaled = self.scaler.transform(X)                         # Scale features using the same scaler as training
        
        return float(self.model.predict_proba(X_scaled)[0][1])      # Return probability of class 1 (success) as float

    def generate_signal(self, row):
        """Rule proposes direction; LR must confirm with P(success) > threshold."""
        direction = super().generate_signal(row)
        if direction == 0:
            return 0
        if self.model is None or self.scaler is None:
            return direction
        p = self.get_probability(row, direction)
        if p > self.CONFIRM_THRESHOLD:
            return direction
        return 0
