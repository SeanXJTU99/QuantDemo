import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ---------- 1. 读 csv ----------
df = pd.read_csv('ashareOutPut.csv')
df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
df = df.sort_values('trade_date').reset_index(drop=True)

# ---------- 2. 特征/标签构造 ----------
df['MA5'] = df['close'].rolling(5).mean()
# 次日对数收益
df['log_ret'] = np.log(df['close']).diff().shift(-1)

# 特征与标签
feature_cols = ['open', 'high', 'low', 'close', 'MA5']
features = df[feature_cols]
label = df[['log_ret']]

# 丢掉 NA（MA5 和 log_ret 产生的）
mask = features.notnull().all(1) & label.notnull().all(1)
features = features[mask]
label = label[mask]
df_clean = df[mask].copy()          # 对齐的干净数据

# ---------- 3. 转 Tensor ----------
X = torch.tensor(features.values, dtype=torch.float32)
y = torch.tensor(label.values, dtype=torch.float32)

# ---------- 4. 网络、损失、优化器 ----------
class RegressionModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

model = RegressionModel(X.shape[1])
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ---------- 5. 训练 ----------
for epoch in range(1, 101):
    pred = model(X)
    loss = criterion(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch:3d} | Loss: {loss.item():.6f}')

# ---------- 6. 预测 & 还原价格 ----------
model.eval()
with torch.no_grad():
    pred_log_ret = model(X).numpy().squeeze()          # 次日收益
true_close = df_clean['close'].values
pred_close = true_close * np.exp(pred_log_ret)         # 还原明日价格

# ---------- 7. 画图 ----------
dates = df_clean['trade_date'].values
plt.figure(figsize=(10, 5))
plt.plot(dates, true_close, label='True Close', color='black')
plt.plot(dates, pred_close, label='Pred Next-Close', color='red', alpha=0.8)
plt.title('SZ000001  Next-Day Close Prediction')
plt.xlabel('trade_date')
plt.ylabel('price')
plt.legend()
plt.tight_layout()
plt.savefig('next_close_pred.png', dpi=300, bbox_inches='tight')
plt.show()

# ================= 8. 交易回测 =================
def trade_by_pred(dates, true_close, pred_close, init_cash=100000):
    """
    简规则：预测价 > 今日收盘 → 开盘做多；预测价 < 今日收盘 → 开盘做空（纯多头版只清仓）
    为了只用现有数据，用今日收盘代替明日开盘价（实盘可换开盘价字段）
    """
    cash, pos = init_cash, 0          # 仓位按股数计
    nav = []                          # 每日净资产

    for t in range(len(dates)-1):     # 最后一天无法交易
        if pred_close[t] > true_close[t]:   # 预测涨 → 满仓
            if pos == 0:                      # 空仓则买入
                pos = cash / true_close[t]
                cash = 0
        else:                               # 预测跌 → 空仓
            if pos > 0:                       # 有仓则卖出
                cash = pos * true_close[t]
                pos = 0

        # 当日收盘净资产
        nav.append(cash + pos * true_close[t])

    # 最后一日按收盘结算
    nav.append(cash + pos * true_close[-1])
    return np.array(nav)

nav = trade_by_pred(dates, true_close, pred_close)

# 评价指标
def evaluate(nav):
    ret = pd.Series(nav).pct_change().dropna()
    annual_ret = ret.mean() * 252
    annual_vol = ret.std() * np.sqrt(252)
    sharpe = annual_ret / annual_vol if annual_vol != 0 else 0
    drawdown = (nav / np.maximum.accumulate(nav) - 1).min()
    print(f'年化收益: {annual_ret:.2%}')
    print(f'年化波动: {annual_vol:.2%}')
    print(f'夏普比率: {sharpe:.2f}')
    print(f'最大回撤: {drawdown:.2%}')

evaluate(nav)

# 资金曲线图
plt.figure(figsize=(10,4))
plt.plot(dates, nav / nav[0], label='Strategy NAV', color='orange')
plt.plot(dates, true_close / true_close[0], label='Buy & Hold', color='gray', alpha=0.7)
plt.title('Strategy vs Buy&Hold')
plt.ylabel('normalised value')
plt.legend()
plt.tight_layout()
plt.savefig('strategy_nav.png', dpi=300, bbox_inches='tight')
plt.show()