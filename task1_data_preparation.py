"""
Task 1: 数据准备与预处理
- 筛选机场口岸数据 (Control_Point = 'Airport')
- 筛选时间范围 (2023.2.6 - 2025.12.17)
- 聚合每日总客流 (Arrival + Departure)
- 分离训练集和测试集 (保留最后5天用于验证)
"""
import pandas as pd
import numpy as np
import os

# 设置路径
data_dir = r"c:\Users\Lenovo\Desktop\HKU MDASC\1. Sem1\8003\project\data"
figures_dir = r"c:\Users\Lenovo\Desktop\HKU MDASC\1. Sem1\8003\project\figures"
os.makedirs(figures_dir, exist_ok=True)

print("="*80)
print("Task 1: 数据准备与预处理")
print("="*80)

# 1. 读取原始数据
print("\n📥 读取原始数据...")
df = pd.read_csv(os.path.join(data_dir, "immd_daily_passenger_clean.csv"))
df['Date'] = pd.to_datetime(df['Date'])
print(f"   原始数据: {len(df):,} 条记录")

# 2. 筛选机场数据
print("\n✂️ 筛选机场口岸数据...")
airport_df = df[df['Control_Point'] == 'Airport'].copy()
print(f"   机场数据: {len(airport_df):,} 条记录")

# 3. 筛选时间范围 (2023年1月1日之后 - 自然年开始，便于分析年度季节性)
print("\n📅 筛选时间范围 (2023.1.1 - 2025.12.17)...")
start_date = pd.Timestamp('2023-01-01')
airport_df = airport_df[airport_df['Date'] >= start_date].copy()
print(f"   筛选后: {len(airport_df):,} 条记录")

# 4. 聚合每日总客流 (入境+出境)
print("\n📊 聚合每日总客流...")
daily_traffic = airport_df.groupby('Date').agg({
    'Total': 'sum',
    'HK_Residents': 'sum',
    'Mainland_Visitors': 'sum',
    'Other_Visitors': 'sum'
}).reset_index()

daily_traffic.columns = ['Date', 'Total', 'HK_Residents', 'Mainland_Visitors', 'Other_Visitors']
daily_traffic = daily_traffic.sort_values('Date').reset_index(drop=True)

print(f"   每日数据: {len(daily_traffic)} 天")
print(f"   日期范围: {daily_traffic['Date'].min().strftime('%Y-%m-%d')} 至 {daily_traffic['Date'].max().strftime('%Y-%m-%d')}")

# 5. 数据质量检查
print("\n🔍 数据质量检查...")
# 检查日期连续性
date_range = pd.date_range(start=daily_traffic['Date'].min(), end=daily_traffic['Date'].max())
missing_dates = set(date_range) - set(daily_traffic['Date'])
print(f"   缺失日期数: {len(missing_dates)}")
if len(missing_dates) > 0:
    print(f"   缺失日期: {sorted(missing_dates)[:5]}...")  # 显示前5个

# 检查缺失值
print(f"   Total列缺失值: {daily_traffic['Total'].isna().sum()}")
print(f"   Total列零值: {(daily_traffic['Total'] == 0).sum()}")

# 6. 分离训练集和测试集 (保留最后5天)
print("\n✂️ 分离训练集和测试集...")
test_size = 5
train_df = daily_traffic.iloc[:-test_size].copy()
test_df = daily_traffic.iloc[-test_size:].copy()

print(f"   训练集: {len(train_df)} 天 ({train_df['Date'].min().strftime('%Y-%m-%d')} 至 {train_df['Date'].max().strftime('%Y-%m-%d')})")
print(f"   测试集: {len(test_df)} 天 ({test_df['Date'].min().strftime('%Y-%m-%d')} 至 {test_df['Date'].max().strftime('%Y-%m-%d')})")

# 7. 添加时间特征
print("\n🕐 添加时间特征...")
for df_temp in [daily_traffic, train_df, test_df]:
    df_temp['Year'] = df_temp['Date'].dt.year
    df_temp['Month'] = df_temp['Date'].dt.month
    df_temp['Day'] = df_temp['Date'].dt.day
    df_temp['DayOfWeek'] = df_temp['Date'].dt.dayofweek  # 0=Monday, 6=Sunday
    df_temp['WeekOfYear'] = df_temp['Date'].dt.isocalendar().week
    df_temp['IsWeekend'] = df_temp['DayOfWeek'].isin([5, 6]).astype(int)

# 8. 保存数据
print("\n💾 保存数据...")
# 完整数据
full_path = os.path.join(data_dir, "airport_daily_traffic.csv")
daily_traffic.to_csv(full_path, index=False)
print(f"   完整数据: {full_path}")

# 训练集
train_path = os.path.join(data_dir, "airport_train.csv")
train_df.to_csv(train_path, index=False)
print(f"   训练集: {train_path}")

# 测试集
test_path = os.path.join(data_dir, "airport_test.csv")
test_df.to_csv(test_path, index=False)
print(f"   测试集: {test_path}")

# 9. 数据摘要
print("\n" + "="*80)
print("📋 数据摘要")
print("="*80)
print(f"""
数据集: 香港机场每日过境旅客总数
来源: 香港入境事务处 (IMMD)
URL: https://www.immd.gov.hk/opendata/eng/transport/immigration_clearance/statistics_on_daily_passenger_traffic.csv

时间范围: {train_df['Date'].min().strftime('%Y-%m-%d')} 至 {test_df['Date'].max().strftime('%Y-%m-%d')}
总样本数: {len(daily_traffic)} 天
训练集: {len(train_df)} 天
测试集: {len(test_df)} 天 (用于最终预测验证)

基本统计量 (训练集):
  均值: {train_df['Total'].mean():,.0f} 人/天
  标准差: {train_df['Total'].std():,.0f}
  最小值: {train_df['Total'].min():,.0f}
  最大值: {train_df['Total'].max():,.0f}
  中位数: {train_df['Total'].median():,.0f}
  25%分位: {train_df['Total'].quantile(0.25):,.0f}
  75%分位: {train_df['Total'].quantile(0.75):,.0f}
""")

# 10. 显示测试集数据 (用于后续验证)
print("\n📊 测试集数据 (最后5天，用于预测验证):")
print(test_df[['Date', 'Total', 'DayOfWeek']].to_string(index=False))

print("\n✅ Task 1 完成!")
