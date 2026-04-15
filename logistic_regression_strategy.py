#logistic_regression_strategy.py
# Standalone rule-based trading strategy enhanced with logistic regression

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

from rule_based_strategy import RuleBasedStrategy


class LogisticRegressionStrategy(RuleBasedStrategy):
    INDICATOR_COLUMNS = ["price_to_sma", "sma_cross", "rsi", "obv_diff"]
    FEATURE_COLUMNS = INDICATOR_COLUMNS + ["signal_sign"]  # Add signal sign as a feature for probability estimation
    REG_C = 4.64  # Default regularization strength, will be tuned on validation data
    # PENALTY = "l1" TODO: Ta bort?
    SOLVER = "liblinear"
    CONFIRMATION_THRESHOLD = 0.50  # Minimum probability to confirm a buy signal, can be tuned on validation data

    def __init__(self, model=None, scaler=None):
        self.model = model
        self.scaler = scaler
        # Set in _prepare_data: indicator columns only, or indicators + signal_sign (same order as training).
        self._fit_columns = None

    def _prepare_data(self, train_data, val_data):
        """Prepare and scale data for training and validation."""

        # Decide which feature columns to use for training and validation.
        available = [c for c in self.FEATURE_COLUMNS if c in train_data.columns]    # "Ideal" feature list (indicator columns + signal_sign)
       
       # If signal_sign isnt available -> train with indicators only
        if not available:
            available = list(self.INDICATOR_COLUMNS)                                # Fall back to indicator columns if signal_sign is not available
        self._fit_columns = available      

        # Prepare X and y for training and validation, using the decided feature columns.
        x_train = train_data[self._fit_columns]     # indicators + signal_sign if available, otherwise only indicators
        y_train = train_data["target"]              # Target always the same (next hour up or not), regardless of features used
        x_val = val_data[self._fit_columns]         # indicators + signal_sign if available, otherwise only indicators
        y_val = val_data["target"]                  # Target always the same (next hour up or not), regardless of features used

        # Scale features
        self.scaler = StandardScaler()                      # Create a new scaler for this training session
        x_train_scaled = self.scaler.fit_transform(x_train) # Fit scaler on training data and transform
        x_val_scaled = self.scaler.transform(x_val)         # Transform validation data using the same scaler 

        # Return prepared and scaled data for training and validation
        return x_train_scaled, y_train, x_val_scaled, y_val

    def tune(self, train_data, val_data):
        """Tune C parameter on validatation data"""
        
        x_train_scaled, y_train, x_val_scaled, y_val = self._prepare_data(train_data, val_data)

        #C_values = [0.01, 0.1, 1, 10, 100]
        C_values = np.logspace(-2, 2, 10).tolist()
        best_C = None
        best_accuracy = 0

        print("\n--- Tuning C parameter ---")
        for C in C_values:
            model = LogisticRegression(
                class_weight="balanced",
                l1_ratio=1,
                #penalty=self.PENALTY, TODO: Ta bort?
                solver=self.SOLVER,
                random_state=42,
                C=C
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
            #penalty=self.PENALTY, TODO: Ta bort?
            solver=self.SOLVER,
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
        for feature, coef in zip(self._fit_columns, self.model.coef_[0]):
            print(f"  {feature}: {coef}")
        print(f"Scaler means: {self.scaler.mean_}")
        print(f"Scaler stds: {self.scaler.scale_}")

        proba = self.model.predict_proba(x_val_scaled)[:, 1]
        print(f"Min sannolikhet:   {proba.min():.3f}")
        print(f"Max sannolikhet:   {proba.max():.3f}")
        print(f"Medel sannolikhet: {proba.mean():.3f}")

    def tune_confirmation_threshold(self, val_data, thresholds=None, min_trades=30):
        """Tune confirmation threshold on validation data using trading metrics.
            Not LR-specific"""
        
        # Import to keep the strategy reusable and avoid module-level coupling
        from backtest import run_backtest
        from backtest_result import BacktestResult

        # If thresholds is not provided, use a default range from 0.45 to 0.60 with 0.01 steps
        if thresholds is None:
            thresholds = [round(x,2) for x in [0.45, 0.46, 0.47, 0.48, 0.49, 0.50, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.57, 0.58, 0.59, 0.60]]

        best_threshold = None
        best_tuple = None       # (accuracy, win_rate, expectancy, profit_factor) for best threshold

        print("\n--- Tuning confirmation threshold ---")
        for t in thresholds:
            self.CONFIRMATION_THRESHOLD = t
            trades_val = run_backtest(self, val_data)
            if trades_val is None or len(trades_val) < min_trades:
                trades_n = 0 if trades_val is None else len(trades_val)
                print(f"{t:<6.2f} {trades_n:<8} {'-':<8} {'-':<10} {'-':<11} {'-':<9}")
                continue

            result = BacktestResult(trades_val, f"Threshold {t:.2f}")
            print(f"{t:<6.2f} {result.total_trades:<8} {result.sharpe_ratio:<8.2f} {result.max_drawdown:<10.2f} {result.expectancy:<11.2f} {result.profit_factor:<9.2f}")

            score = (result.sharpe_ratio, result.expectancy, result.max_drawdown)  # Example tuple of metrics to compare
            if best_tuple is None:
                best_tuple = score
                best_threshold = t
            else:
                # Primary...
                if (score[0] > best_tuple[0] #...Sharpe higher
                    or (score[0] == best_tuple[0] and score[1] > best_tuple[1]) #...Tie-break: expectancy higher
                    or (score[0] == best_tuple[0] and score[1] == best_tuple[1] and score[2] > best_tuple[2])):     #...Tie-break: drawdown less negative (higher)
                    best_tuple = score
                    best_threshold = t
            
            if best_threshold is not None:
                self.CONFIRMATION_THRESHOLD = best_threshold
                print(f"\nChosen CONFIRMATION_THRESHOLD: {best_threshold:.2f} (min_trades={min_trades})")
            else:
                print(f"\nNo threshold gave >= {min_trades} trades on validation data (val_data): keep default: {self.CONFIRMATION_THRESHOLD:.2f}")

            return best_threshold


    def get_probability(self, row, signal):
        """Same idea as before: one row of features -> scale -> proba; plus signal_sign when the model was trained with it."""
        if self._fit_columns is None:
            raise ValueError("Model is not trained or fit columns are not defined.")

        sign = 1.0 if signal == 1 else -1.0
        row_values = {}
        for col in self._fit_columns:
            if col == "signal_sign":
                row_values[col] = sign
            else:
                row_values[col] = float(row[col])

        # One row, same columns as training (like row[INDICATOR_COLUMNS].to_frame().T, extended if needed).
        latest = pd.DataFrame([row_values], columns=self._fit_columns)
        latest_scaled = self.scaler.transform(latest)
        probability = self.model.predict_proba(latest_scaled)[0, 1]
        return float(probability)

    def generate_signal(self, row):
        
        direction = super().generate_signal(row)
        if direction == 0:
            return 0  # Hold signal, no need to calculate probability
        if self.model is None or self.scaler is None:
            return direction  # If model isn't trained, fall back to rule-based signal
        
        p = self.get_probability(row, direction)

        if p > self.CONFIRMATION_THRESHOLD:
            return direction  # Return original buy/sell signal if probability is high enough
        return 0  # Otherwise, return hold signal
