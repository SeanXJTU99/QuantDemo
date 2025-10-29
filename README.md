# QuantDemo
10.29
Daily-frequency, directional, single-asset (000001.SZ) short-term momentum/reversal quantitative CTA, feed-forward fully-connected neural network (MLP), price-momentum/trend category without fundamentals, without volatility, without volume factors; (public-quant) annualized return 5.96%, max drawdown 31%, Sharpe 0.25—almost identical to mine, but that fund is a mixed equity-long-only product; my strategy should have had a lower-drawdown advantage, yet the result is higher, indicating insufficient risk control.

Next improvements:

    Expand the factor library: add turnover, volatility, technical indicators (RSI, MACD), industry/style factors.
    Introduce a risk model: stop-loss, position sizing, volatility weighting.
    Change the model: LightGBM or LSTM, which can usually raise Sharpe to 0.8+.
    Multi-asset universe: single-stock noise is large; diversifying into 50-300 stocks or ETFs can significantly reduce drawdown.
    Out-of-sample/rolling training: avoid over-fitting and improve robustness.


10.28
a simple demo for daily quant using Pytorch, involving data aggregation and cleaning, feature engineering, model construction and training, prediction and evaluation

data from tushare for free


