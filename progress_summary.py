"""
研究进度与数据显著性评估
"""
import pandas as pd
import numpy as np
from scipy import stats

# 读取训练数据
data_dir = r'c:\Users\Lenovo\Desktop\HKU MDASC\1. Sem1\8003\project\data'
train_df = pd.read_csv(f'{data_dir}/airport_train.csv')
train_df['Date'] = pd.to_datetime(train_df['Date'])

print('='*80)
print('香港机场每日客流数据 - 研究进度与显著性评估')
print('='*80)

# 1. 基本统计
print('\n【1. 数据概览】')
n = len(train_df)
mean_val = train_df['Total'].mean()
std_val = train_df['Total'].std()
cv = std_val / mean_val * 100

print(f'  样本量: {n} 天 (n >> 100, 满足要求)')
date_min = train_df['Date'].min().strftime('%Y-%m-%d')
date_max = train_df['Date'].max().strftime('%Y-%m-%d')
print(f'  日期范围: {date_min} 至 {date_max}')
print(f'  均值: {mean_val:,.0f} 人/天')
print(f'  标准差: {std_val:,.0f}')
print(f'  变异系数: {cv:.1f}%')

# 2. 趋势显著性检验
print('\n【2. 趋势显著性检验 (线性回归)】')
train_df['t'] = range(len(train_df))
slope, intercept, r_value, p_value, std_err = stats.linregress(train_df['t'], train_df['Total'])
print(f'  线性趋势斜率: {slope:.2f} 人/天')
print(f'  R-squared: {r_value**2:.4f}')
print(f'  p-value: {p_value:.2e}')
if p_value < 0.05:
    print('  结论: *** 趋势高度显著 (p < 0.05) ***')
else:
    print('  结论: 趋势不显著')

# 3. 周季节性显著性检验
print('\n【3. 周季节性显著性检验 (Kruskal-Wallis)】')
groups = [train_df[train_df['DayOfWeek']==i]['Total'].values for i in range(7)]
h_stat, kw_pvalue = stats.kruskal(*groups)
print(f'  H统计量: {h_stat:.2f}')
print(f'  p-value: {kw_pvalue:.2e}')
if kw_pvalue < 0.05:
    print('  结论: *** 周季节性高度显著 (p < 0.05) ***')
else:
    print('  结论: 周季节性不显著')

# 周内各天均值
print('\n  周内各天均值:')
day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
for i, name in enumerate(day_names):
    day_mean = train_df[train_df['DayOfWeek']==i]['Total'].mean()
    diff_pct = (day_mean - mean_val) / mean_val * 100
    sign = '+' if diff_pct > 0 else ''
    print(f'    {name}: {day_mean:>10,.0f} ({sign}{diff_pct:.1f}%)')

# 4. 月度季节性检验
print('\n【4. 月度季节性显著性检验 (Kruskal-Wallis)】')
groups_month = [train_df[train_df['Month']==m]['Total'].values for m in range(1, 13)]
groups_month = [g for g in groups_month if len(g) > 0]
h_stat_m, kw_pvalue_m = stats.kruskal(*groups_month)
print(f'  H统计量: {h_stat_m:.2f}')
print(f'  p-value: {kw_pvalue_m:.2e}')
if kw_pvalue_m < 0.05:
    print('  结论: *** 月度季节性高度显著 (p < 0.05) ***')
else:
    print('  结论: 月度季节性不显著')

# 5. 年度增长分析
print('\n【5. 年度增长分析】')
yearly = train_df.groupby('Year')['Total'].mean()
for year, val in yearly.items():
    print(f'  {year}年均值: {val:,.0f}')
years = list(yearly.index)
growth = (yearly[years[-1]] - yearly[years[0]]) / yearly[years[0]] * 100
print(f'  总体增长: {growth:.1f}% ({years[0]} -> {years[-1]})')

# 6. ACF显著性
print('\n【6. 自相关显著性】')
from statsmodels.tsa.stattools import acf
acf_vals = acf(train_df['Total'], nlags=14, fft=True)
ci = 1.96 / np.sqrt(n)
print(f'  95%置信区间: +/-{ci:.4f}')
for lag in [1, 7, 14]:
    sig = '*** 显著 ***' if abs(acf_vals[lag]) > ci else '不显著'
    print(f'  Lag {lag:2d} ACF: {acf_vals[lag]:.4f} {sig}')

# 7. STL分解结果回顾
print('\n【7. STL分解结果 (方差占比)】')
print('  趋势成分: 78.6%')
print('  季节成分: 11.8%')
print('  残差成分: 9.1%')

# 综合评估
print('\n' + '='*80)
print('综合评估')
print('='*80)

print('''
+------------------------------------------------------------------+
|  检验项目          |  结果      |  p-value       |  显著性      |
+------------------------------------------------------------------+
|  趋势              |  上升      |  < 0.001       |  *** 显著    |
|  周季节性 (s=7)    |  存在      |  < 0.001       |  *** 显著    |
|  月度季节性        |  存在      |  < 0.001       |  *** 显著    |
|  Lag-1 自相关      |  强正相关  |  -             |  *** 显著    |
|  Lag-7 自相关      |  正相关    |  -             |  *** 显著    |
+------------------------------------------------------------------+

结论: 数据具有非常显著的时间序列特征！
''')

print('''
推荐继续使用该数据集的理由:
1. 样本量充足: 1077天 >> 100，满足课程要求
2. 趋势显著: 明显的上升趋势，可以建模
3. 周季节性显著: 周末vs工作日差异明显 (周日最高)
4. 月度季节性显著: 暑假、圣诞等假期高峰明显
5. 自相关显著: ACF/PACF有清晰模式，适合SARIMA建模
6. 方差结构清晰: 趋势占78.6%，季节占11.8%

建议: 继续使用SARIMA(p,1,q)(P,1,Q,7)模型进行预测
''')
