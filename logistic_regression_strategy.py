#logistic_regression_strategy.py
# Standalone rule-based trading strategy enhanced with logistic regression

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from data_pipeline import split_data
import numpy as np
import pandas as pd

#import joblib TODO: Delete or save?

from rule_based_strategy import RuleBasedStrategy

class LogisticRegressionStrategy(RuleBasedStrategy):
    INDICATOR_COLUMNS = ["price_to_sma", "sma_cross", "rsi", "obv_diff"]
    FEATURE_COLUMNS = INDICATOR_COLUMNS + ["signal_sign"]  # Add signal sign as a feature for probability estimation
    REG_C = 0.64  # Default regularization strength, will be tuned on validation data
    # PENALTY = "l1" TODO: Ta bort?
    SOLVER = "liblinear"
    CONFIRMATION_THRESHOLD = 0.47  # Minimum probability to confirm a buy signal, can be tuned on validation data

    # Constructor
    def __init__(self, model=None, scaler=None):
        self.model = model
        self.scaler = scaler
        #TODO: NEW
        # Columns used for fitting the logistic regression model, set during training
        self.fit_columns = None

    def _prepare_data(self, train_data, val_data):
        # Long-only training often omits signal_sign; long+short uses _build_rule_trading_split which adds it.
        available = [c for c in self.FEATURE_COLUMNS if c in train_data.columns]
        if not available:
            available = list(self.INDICATOR_COLUMNS)
        self._fit_columns = available

        x_train = train_data[self._fit_columns]
        y_train = train_data["target"]
        x_val = val_data[self._fit_columns]
        y_val = val_data["target"]

        self.scaler = StandardScaler()
        x_train_scaled = self.scaler.fit_transform(x_train)
        x_val_scaled = self.scaler.transform(x_val)

        return x_train_scaled, y_train, x_val_scaled, y_val

        

    """def _prepare_data(self, train_data, val_data):
        Prepare and scale features for training and validation.
        
        Returns scaled train and validation sets.
        
        #TODO: NEW
        #x_train = train_data[self.INDICATOR_COLUMNS] TODO: deleted
        x_train = train_data[self.FEATURE_COLUMNS]
        available = [c for c in self.FEATURE_COLUMNS if c in train_data.columns]
        if not available: 
            available = list(self.INDICATOR_COLUMNS)  # Fall back to indicator columns if signal_sign is not available]
        self.fit_columns = available     #TODO: NEW

        x_train = train_data[self.fit_columns]   #TODO: NEW
        y_train = train_data["target"]
        #x_val = val_data[self.INDICATOR_COLUMNS]
        x_val = val_data[self._fit_columns]  #TODO: NEW
        y_val = val_data["target"]

        self.scaler = StandardScaler()
        x_train_scaled = self.scaler.fit_transform(x_train)
        x_val_scaled = self.scaler.transform(x_val)

        return x_train_scaled, y_train, x_val_scaled, y_val
    """

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
        for feature, coef in zip(self._fit_columns, self.model.coef_[0]):       #TODO: new - use self._fit_columns instead of self.INDICATOR_COLUMNS
            print(f"  {feature}: {coef}")
        #for feature, coef in zip(self.INDICATOR_COLUMNS, self.model.coef_[0]): TODO: deleted
        #    print(f"  {feature}: {coef}")  
        print(f"Scaler means: {self.scaler.mean_}")
        print(f"Scaler stds: {self.scaler.scale_}")

        proba = self.model.predict_proba(x_val_scaled)[:, 1]
        print(f"Min sannolikhet:   {proba.min():.3f}")
        print(f"Max sannolikhet:   {proba.max():.3f}")
        print(f"Medel sannolikhet: {proba.mean():.3f}")
   
   # TODO: new
    def get_probability(self, row, signal):

        if self._fit_columns is None:
            raise ValueError("Model is not trained or fit columns are not defined.")
        
        sign = 1.0 if signal == 1 else -1.0
        data = {}
        for c in self._fit_columns:
            if c == "signal_sign":
                data[c] = sign
            else:
                data[c] = float(row[c])
        
        x = pd.DataFrame([data], columns=self._fit_columns)
        x_scaled = self.scaler.transform(x)

        return float(self.model.predict_proba(x_scaled)[0][1])  # Probability of class 1 (buy signal)




    #def get_probability(self, row):
    #    """Return raw probability for interval-based backtesting."""
    #    latest = row[self.INDICATOR_COLUMNS].to_frame().T
    #    latest_scaled = self.scaler.transform(latest)
    #    return self.model.predict_proba(latest_scaled)[0][1]

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

    """#def generate_signal(self, row):
            
            NOTE: Not used in current implementation.
            Live trading is handled by the MQL5 Expert Advisor.
            Backtesting uses get_probability() instead.
            Could be used if live trading is moved to Python in the future.
            
            latest = row[self.INDICATOR_COLUMNS].to_frame().T
            latest_scaled = self.scaler.transform(latest)
            probability = self.model.predict_proba(latest_scaled)[0][1]  # Probability of class 1 (buy signal)

            if probability > 0.52:
                return 1    # Buy signal
            elif probability < 0.47:
                return -1   # Sell signal
            else:
                return 0    # Hold signal"""
