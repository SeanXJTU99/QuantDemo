import tushare as ts
import pandas as pd
import pyarrow as pa
import torch
import torch.nn as nn
import torch.optim as opt
import matplotlib.pyplot as plt
# get daily share data with tushare
pro = ts.pro_api('5a127d297697121031f3df8e550fefcc0db5773557fbf8a968e27b1e')
data1 = pro.daily(ts_code='000001.SZ', start_date='20200101', end_date='20241231')
data2 = pro.daily(ts_code='000001.SZ', start_date='20180701', end_date='20180718')
# save data file
data1.to_parquet("ashare_data.parquet")
#df = pd.read_parquet('ashare_data.parquet', schema=schema)
# read
shareData = pd.read_parquet('ashare_data.parquet')

csv_file_path = 'ashareOutPut.csv'
shareData.to_csv(csv_file_path, index=False)
# 1. 读文件
df = pd.read_csv('ashareOutPut.csv')          # 换成自己的文件名

# 2. 把 trade_date 转成真正的日期
df['trade_date'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')

# 3. 按日期升序排（避免折线来回跳）
df = df.sort_values('trade_date')

# 4. 画图
cols_to_plot = df.columns[2:6]             # 第 3–8 列（open … change）
plt.figure(figsize=(10, 5))
for col in cols_to_plot:
    plt.plot(df['trade_date'], df[col], label=col)

plt.xlabel('trade_date')
plt.ylabel('price / change')
plt.title('SZ000001 daily')
plt.legend()
plt.tight_layout()
# 4. 保存为图片（支持 png/pdf/svg/jpg...）
plt.savefig('000001_daily.png',            # 输出文件名
            dpi=300,                       # 分辨率
            bbox_inches='tight',           # 去掉白边
            facecolor='white')             # 背景白色

plt.show()

# feature engineering
shareData['MA5'] = shareData['close'].rolling(window=5).mean()  #5日均线

# clean
shareData.dropna(inplace=True)

# 删除非数值列
shareData = shareData.select_dtypes(include=['number'])
tensor_data = torch.tensor(shareData.values, dtype=torch.float32)
#print(tensor_data)

class TushareNet(nn.Module):
    def __init__(self):
        super(TushareNet, self).__init__()
        self.fc = nn.Linear(10,1) #全连接层，输入唯10，输出维度1

    def forward(self, x):
        x = self.fc(x)
        return x

model = TushareNet()
print(f'TushareNet model initiative:\n',model)

#损失函数和优化器
crit = nn.MSELoss()
optim = opt.Adam(model.parameters(), lr=0.001)

# 初始化训练数据
inputs = torch.randn(100,10)
labels = torch.randn(100,1)

# training
for epoch in range(100):
    # forward expand
    outputs = model(inputs)
    loss = crit(outputs, labels)

    # reverse expand & optimize
    optim.zero_grad()
    loss.backward()
    optim.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch[{epoch+1}], Loss:{loss.item():.4f}')

# evaluation


