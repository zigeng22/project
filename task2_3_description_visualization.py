"""
Task 2-3: Q1数据描述 + Q2时间序列可视化
- 数据描述和基本统计量
- 时间序列图和特征识别
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

# 路径设置
data_dir = r"c:\Users\Lenovo\Desktop\HKU MDASC\1. Sem1\8003\project\data"
figures_dir = r"c:\Users\Lenovo\Desktop\HKU MDASC\1. Sem1\8003\project\figures"
os.makedirs(figures_dir, exist_ok=True)

# 读取数据
print("="*80)
print("Task 2-3: Q1数据描述 + Q2时间序列可视化")
print("="*80)

train_df = pd.read_csv(os.path.join(data_dir, "airport_train.csv"))
train_df['Date'] = pd.to_datetime(train_df['Date'])

test_df = pd.read_csv(os.path.join(data_dir, "airport_test.csv"))
test_df['Date'] = pd.to_datetime(test_df['Date'])

full_df = pd.read_csv(os.path.join(data_dir, "airport_daily_traffic.csv"))
full_df['Date'] = pd.to_datetime(full_df['Date'])

# ============================================================
# Q1: 数据描述
# ============================================================
print("\n" + "="*80)
print("Q1: 数据描述 (Data Description)")
print("="*80)

print("""
【数据来源】
数据集: 香港机场每日过境旅客统计
来源: 香港入境事务处 (Immigration Department, IMMD)
URL: https://www.immd.gov.hk/opendata/eng/transport/immigration_clearance/statistics_on_daily_passenger_traffic.csv

【选择理由】
1. 样本量充足: 1,082天日度数据 (n >> 100)
2. 时间序列特征丰富: 包含趋势、周季节性、年季节性
3. 数据质量高: 无缺失值，官方数据来源可靠
4. 实际应用价值: 可用于机场运营规划、人力配置

【数据背景】
- 香港国际机场是亚洲主要航空枢纽之一
- 2023年1月8日: 香港与内地恢复首阶段通关
- 2023年2月6日: 全面恢复通关，取消配额限制
- 数据反映了后疫情时代航空出行的恢复趋势
""")

# 基本统计量
print("\n【基本统计量】")
stats = train_df['Total'].describe()
print(f"  样本数 (n):     {len(train_df)}")
print(f"  均值 (Mean):    {stats['mean']:,.0f}")
print(f"  标准差 (Std):   {stats['std']:,.0f}")
print(f"  最小值 (Min):   {stats['min']:,.0f}")
print(f"  25%分位数:      {stats['25%']:,.0f}")
print(f"  中位数 (Median):{stats['50%']:,.0f}")
print(f"  75%分位数:      {stats['75%']:,.0f}")
print(f"  最大值 (Max):   {stats['max']:,.0f}")
print(f"  变异系数 (CV):  {stats['std']/stats['mean']*100:.1f}%")

# 按年统计
print("\n【年度统计】")
yearly = train_df.groupby('Year')['Total'].agg(['mean', 'std', 'min', 'max', 'count'])
for year, row in yearly.iterrows():
    print(f"  {year}年: 均值={row['mean']:,.0f}, 标准差={row['std']:,.0f}, 天数={row['count']:.0f}")

# 按星期统计
print("\n【周内模式】")
day_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
dow_stats = train_df.groupby('DayOfWeek')['Total'].mean()
for dow, val in dow_stats.items():
    print(f"  {day_names[dow]}: {val:,.0f}")

# ============================================================
# Q2: 时间序列可视化
# ============================================================
print("\n" + "="*80)
print("Q2: 时间序列可视化 (Time Series Plot)")
print("="*80)

# 图1: 主时间序列图
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# 1.1 原始时间序列
ax1 = axes[0]
ax1.plot(train_df['Date'], train_df['Total'], linewidth=0.8, color='steelblue', alpha=0.8)
ax1.axhline(y=train_df['Total'].mean(), color='red', linestyle='--', linewidth=1, label=f'Mean: {train_df["Total"].mean():,.0f}')

# 标注关键事件
events = [
    ('2023-01-08', 'Partial\nReopening'),
    ('2023-02-06', 'Full\nReopening'),
    ('2023-01-22', 'CNY'),
    ('2024-02-10', 'CNY'),
    ('2025-01-29', 'CNY'),
]
for date_str, label in events:
    date = pd.Timestamp(date_str)
    if date >= train_df['Date'].min() and date <= train_df['Date'].max():
        ax1.axvline(x=date, color='green', linestyle=':', alpha=0.5)
        ax1.annotate(label, xy=(date, train_df['Total'].max()*0.95), fontsize=8, ha='center')

ax1.set_title('Hong Kong Airport Daily Passenger Traffic (2023-2025)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Date')
ax1.set_ylabel('Daily Passengers')
ax1.legend(loc='lower right')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
ax1.set_ylim(0, train_df['Total'].max() * 1.1)

# 1.2 月度聚合趋势
ax2 = axes[1]
monthly = train_df.groupby(train_df['Date'].dt.to_period('M'))['Total'].agg(['mean', 'std'])
monthly.index = monthly.index.to_timestamp()
ax2.plot(monthly.index, monthly['mean'], marker='o', linewidth=2, markersize=4, color='steelblue')
ax2.fill_between(monthly.index, 
                  monthly['mean'] - monthly['std'], 
                  monthly['mean'] + monthly['std'], 
                  alpha=0.2, color='steelblue')
ax2.set_title('Monthly Average with Standard Deviation Band', fontsize=12)
ax2.set_xlabel('Month')
ax2.set_ylabel('Monthly Average Passengers')
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

# 1.3 周内模式 (箱线图)
ax3 = axes[2]
dow_data = [train_df[train_df['DayOfWeek'] == i]['Total'].values for i in range(7)]
day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
bp = ax3.boxplot(dow_data, tick_labels=day_labels, patch_artist=True)
colors = ['#f0f0f0']*5 + ['#a6cee3', '#a6cee3']  # 周末用不同颜色
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
ax3.set_title('Weekly Pattern (Box Plot by Day of Week)', fontsize=12)
ax3.set_xlabel('Day of Week')
ax3.set_ylabel('Daily Passengers')
ax3.axhline(y=train_df['Total'].mean(), color='red', linestyle='--', linewidth=1, alpha=0.5)

plt.tight_layout()
fig_path1 = os.path.join(figures_dir, "01_time_series_overview.png")
plt.savefig(fig_path1, dpi=150, bbox_inches='tight')
print(f"\n✅ 图1已保存: {fig_path1}")

# 图2: 季节性子图
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))

# 2.1 按月份的季节性
ax = axes2[0, 0]
monthly_pattern = train_df.groupby('Month')['Total'].mean()
ax.bar(range(1, 13), monthly_pattern.values, color='steelblue', alpha=0.8)
ax.axhline(y=train_df['Total'].mean(), color='red', linestyle='--', linewidth=1)
ax.set_title('Monthly Seasonality Pattern', fontsize=12)
ax.set_xlabel('Month')
ax.set_ylabel('Average Daily Passengers')
ax.set_xticks(range(1, 13))
ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

# 2.2 按年对比
ax = axes2[0, 1]
for year in train_df['Year'].unique():
    year_data = train_df[train_df['Year'] == year].copy()
    year_data['DayOfYear'] = year_data['Date'].dt.dayofyear
    ax.plot(year_data['DayOfYear'], year_data['Total'], label=str(year), alpha=0.7, linewidth=0.8)
ax.set_title('Year-over-Year Comparison', fontsize=12)
ax.set_xlabel('Day of Year')
ax.set_ylabel('Daily Passengers')
ax.legend()
ax.set_xlim(1, 366)

# 2.3 周末vs工作日
ax = axes2[1, 0]
weekend_data = train_df[train_df['IsWeekend'] == 1]['Total']
weekday_data = train_df[train_df['IsWeekend'] == 0]['Total']
ax.hist([weekday_data, weekend_data], bins=30, label=['Weekday', 'Weekend'], 
        color=['steelblue', 'coral'], alpha=0.7, edgecolor='white')
ax.axvline(x=weekday_data.mean(), color='steelblue', linestyle='--', linewidth=2, label=f'Weekday Mean: {weekday_data.mean():,.0f}')
ax.axvline(x=weekend_data.mean(), color='coral', linestyle='--', linewidth=2, label=f'Weekend Mean: {weekend_data.mean():,.0f}')
ax.set_title('Distribution: Weekday vs Weekend', fontsize=12)
ax.set_xlabel('Daily Passengers')
ax.set_ylabel('Frequency')
ax.legend(fontsize=9)

# 2.4 滚动统计
ax = axes2[1, 1]
train_df_sorted = train_df.sort_values('Date')
rolling_mean = train_df_sorted['Total'].rolling(window=7).mean()
rolling_std = train_df_sorted['Total'].rolling(window=7).std()
ax.plot(train_df_sorted['Date'], train_df_sorted['Total'], alpha=0.3, linewidth=0.5, color='gray', label='Daily')
ax.plot(train_df_sorted['Date'], rolling_mean, color='steelblue', linewidth=1.5, label='7-day Rolling Mean')
ax.fill_between(train_df_sorted['Date'], 
                rolling_mean - rolling_std, 
                rolling_mean + rolling_std, 
                alpha=0.2, color='steelblue')
ax.set_title('7-Day Rolling Mean with Std Band', fontsize=12)
ax.set_xlabel('Date')
ax.set_ylabel('Passengers')
ax.legend()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

plt.tight_layout()
fig_path2 = os.path.join(figures_dir, "02_seasonality_analysis.png")
plt.savefig(fig_path2, dpi=150, bbox_inches='tight')
print(f"✅ 图2已保存: {fig_path2}")

# ============================================================
# 时间序列特征总结
# ============================================================
print("\n" + "="*80)
print("时间序列特征总结")
print("="*80)

print("""
【趋势 (Trend)】
- 整体呈上升趋势，反映疫情后航空出行的持续恢复
- 2023年初: 日均约5万人
- 2025年末: 日均超过12万人
- 增长约2.4倍

【季节性 (Seasonality)】
1. 周季节性 (Period = 7):
   - 周六、周日客流明显高于工作日
   - 周末均值约12万，工作日均值约10万
   
2. 年季节性:
   - 暑假 (7-8月): 高峰期
   - 圣诞/新年 (12月-1月): 高峰期
   - 春节: 显著高峰
   - 3-4月、9-10月: 相对低谷

【周期性 (Cyclicity)】
- 无明显的多年周期性（数据跨度仅约3年）

【异常值 (Outliers)】
- 2023年1月初: 客流异常低（尚未全面通关）
- 春节期间: 出现显著高峰
- 部分日期可能受天气、航班取消等影响出现低值
""")

plt.close('all')
print("\n✅ Task 2-3 完成!")
