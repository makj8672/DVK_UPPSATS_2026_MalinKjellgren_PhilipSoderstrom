//+------------------------------------------------------------------+
//| XAU_USD_Strategy.mq5                                             |
//| Rule-based strategy enhanced with Logistic Regression            |
//+------------------------------------------------------------------+
#property copyright "2026"
#property version   "1.00"

#include <Trade\Trade.mqh>
CTrade trade;

//--- Logistic Regression parameters (exported from Python)
double LR_INTERCEPT            =  0.0;
double LR_COEF_PRICE_TO_SMA    = -0.0014775467098251517;
double LR_COEF_SMA_CROSS       = -0.05401952758770533;
double LR_COEF_RSI             = -0.05907902933460125;
double LR_COEF_OBV_DIFF        =  0.0;

//--- Scaler parameters (exported from Python)
double SCALER_MEAN_PRICE_TO_SMA =  1.26173964e-02;
double SCALER_MEAN_SMA_CROSS    =  1.10097394e-02;
double SCALER_MEAN_RSI          =  5.34990870e+01;
double SCALER_MEAN_OBV_DIFF     =  5.12834796e-03;

double SCALER_STD_PRICE_TO_SMA  =  9.30170135e-03;
double SCALER_STD_SMA_CROSS     =  8.06071600e-03;
double SCALER_STD_RSI           =  7.28638368e+00;
double SCALER_STD_OBV_DIFF      =  5.40961690e-03;

//--- Strategy parameters
double STOP_LOSS_PCT    = 1.0;
double TAKE_PROFIT_PCT  = 2.0;
double LOT_SIZE         = 0.01;

//+------------------------------------------------------------------+
//| Calculate sigmoid function                                        |
//+------------------------------------------------------------------+
double Sigmoid(double z)
{
    return 1.0 / (1.0 + MathExp(-z));
}

//+------------------------------------------------------------------+
//| Calculate LR probability                                         |
//+------------------------------------------------------------------+
double GetProbability(double price_to_sma, double sma_cross, double rsi, double obv_diff)
{
    //--- Standardize features
    double x1 = (price_to_sma - SCALER_MEAN_PRICE_TO_SMA) / SCALER_STD_PRICE_TO_SMA;
    double x2 = (sma_cross    - SCALER_MEAN_SMA_CROSS)    / SCALER_STD_SMA_CROSS;
    double x3 = (rsi          - SCALER_MEAN_RSI)          / SCALER_STD_RSI;
    double x4 = (obv_diff     - SCALER_MEAN_OBV_DIFF)     / SCALER_STD_OBV_DIFF;

    //--- Calculate z
    double z = LR_INTERCEPT
             + LR_COEF_PRICE_TO_SMA * x1
             + LR_COEF_SMA_CROSS    * x2
             + LR_COEF_RSI          * x3
             + LR_COEF_OBV_DIFF     * x4;

    return Sigmoid(z);
}

//+------------------------------------------------------------------+
//| Check rule-based conditions                                       |
//+------------------------------------------------------------------+
bool RuleBasedSignal(double price_to_sma, double sma_cross, double rsi, double obv_diff)
{
    bool cond1 = price_to_sma > 0;          // Price over SMA200
    bool cond2 = sma_cross > 0;             // SMA50 over SMA200
    bool cond3 = rsi >= 35 && rsi <= 65;    // RSI neutral
    bool cond4 = obv_diff > 0;              // OBV rising

    return cond1 && cond2 && cond3 && cond4;
}

//+------------------------------------------------------------------+
//| Expert tick function                                              |
//+------------------------------------------------------------------+
void OnTick()
{
    static datetime last_bar = 0;
    datetime current_bar = iTime(_Symbol, PERIOD_H1, 0);
    if(current_bar == last_bar) return;
    last_bar = current_bar;

    if(PositionsTotal() > 0)
    {
        Print("Position öppen - hoppar över signal");
        return;
    }

    int sma50_handle  = iMA(_Symbol, PERIOD_H1, 50,  0, MODE_SMA, PRICE_CLOSE);
    int sma200_handle = iMA(_Symbol, PERIOD_H1, 200, 0, MODE_SMA, PRICE_CLOSE);
    int rsi_handle    = iRSI(_Symbol, PERIOD_H1, 14, PRICE_CLOSE);

    double sma50[], sma200[], rsi_val[];
    ArraySetAsSeries(sma50,   true);
    ArraySetAsSeries(sma200,  true);
    ArraySetAsSeries(rsi_val, true);

    CopyBuffer(sma50_handle,  0, 0, 2, sma50);
    CopyBuffer(sma200_handle, 0, 0, 2, sma200);
    CopyBuffer(rsi_handle,    0, 0, 2, rsi_val);

    int obv_handle = iOBV(_Symbol, PERIOD_H1, VOLUME_TICK);
    double obv[];
    ArraySetAsSeries(obv, true);
    CopyBuffer(obv_handle, 0, 0, 3, obv);

    double close        = iClose(_Symbol, PERIOD_H1, 1);
    double price_to_sma = (close - sma200[1]) / sma200[1];
    double sma_cross    = (sma50[1] - sma200[1]) / sma200[1];
    double rsi          = rsi_val[1];
    double obv_diff     = obv[1] != 0 ? (obv[1] - obv[2]) / MathAbs(obv[2]) : 0;
    double probability  = GetProbability(price_to_sma, sma_cross, rsi, obv_diff);
    bool   signal       = RuleBasedSignal(price_to_sma, sma_cross, rsi, obv_diff);

    Print("--- Ny H1 bar ---");
    Print("price_to_sma=", DoubleToString(price_to_sma, 5),
          " sma_cross=",   DoubleToString(sma_cross, 5),
          " rsi=",         DoubleToString(rsi, 2),
          " obv_diff=",    DoubleToString(obv_diff, 5));
    Print("Probability=",  DoubleToString(probability, 4),
          " Signal=",      signal ? "KÖP" : "HÅLL");

    if(!signal) return;

    Print("Signal generated! Probability: ", DoubleToString(probability, 4));

    double entry = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    double sl    = entry * (1 - STOP_LOSS_PCT / 100);
    double tp    = entry * (1 + TAKE_PROFIT_PCT / 100);

    trade.Buy(LOT_SIZE, _Symbol, entry, sl, tp,
              StringFormat("LR_Strategy p=%.3f", probability));
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                  |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
}