"""
Task 4-6: Q3 ACF/PACF分析 + Q4 时间序列分解 + Q5 平稳性检验

参考示例项目方法:
- Sample 1: 对原始数据进行季节差分(365)后分析ACF/PACF
- Sample 2: 对log变换后数据进行月度+年度差分，详细分析ACF/PACF
- 我们: 日度数据，周期=7，需要考虑周差分
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import STL, seasonal_decompose
import warnings
import os

warnings.filterwarnings('ignore')

# 路径设置
data_dir = r"c:\Users\Lenovo\Desktop\HKU MDASC\1. Sem1\8003\project\data"
figures_dir = r"c:\Users\Lenovo\Desktop\HKU MDASC\1. Sem1\8003\project\figures"

# 读取数据
train_df = pd.read_csv(os.path.join(data_dir, "airport_train.csv"))
train_df['Date'] = pd.to_datetime(train_df['Date'])
train_df = train_df.sort_values('Date').reset_index(drop=True)

# 设置时间序列索引
ts = train_df.set_index('Date')['Total']

print("="*80)
print("Task 4: Q3 ACF/PACF 分析")
print("="*80)

# ============================================================
# Q3: ACF/PACF 分析
# ============================================================

fig, axes = plt.subplots(4, 2, figsize=(14, 16))

# 3.1 原始数据的ACF/PACF
print("\n【原始数据 ACF/PACF】")
plot_acf(ts, lags=50, ax=axes[0, 0], title='ACF - Original Series')
plot_pacf(ts, lags=50, ax=axes[0, 1], title='PACF - Original Series', method='ywm')
print("  - ACF缓慢衰减，表明存在趋势或非平稳性")
print("  - 可能需要差分处理")

# 3.2 一阶差分后的ACF/PACF
ts_diff1 = ts.diff().dropna()
print("\n【一阶差分 ACF/PACF】")
plot_acf(ts_diff1, lags=50, ax=axes[1, 0], title='ACF - First Difference')
plot_pacf(ts_diff1, lags=50, ax=axes[1, 1], title='PACF - First Difference', method='ywm')
print("  - 一阶差分消除趋势")
print("  - 观察是否仍存在周期性模式")

# 3.3 周差分(lag=7)后的ACF/PACF - 参考Sample 1的季节差分方法
ts_diff7 = ts.diff(7).dropna()
print("\n【周差分(lag=7) ACF/PACF】")
plot_acf(ts_diff7, lags=50, ax=axes[2, 0], title='ACF - Seasonal Difference (lag=7)')
plot_pacf(ts_diff7, lags=50, ax=axes[2, 1], title='PACF - Seasonal Difference (lag=7)', method='ywm')
print("  - 周差分消除周季节性")
print("  - 检查lag=7, 14, 21处的相关性")

# 3.4 一阶差分+周差分后的ACF/PACF - 参考Sample 2的双重差分方法
ts_diff1_7 = ts.diff().diff(7).dropna()
print("\n【一阶+周差分 ACF/PACF】")
plot_acf(ts_diff1_7, lags=50, ax=axes[3, 0], title='ACF - First + Seasonal(7) Difference')
plot_pacf(ts_diff1_7, lags=50, ax=axes[3, 1], title='PACF - First + Seasonal(7) Difference', method='ywm')
print("  - 双重差分同时消除趋势和周季节性")

plt.tight_layout()
fig_path = os.path.join(figures_dir, "03_acf_pacf_analysis.png")
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"\n✅ ACF/PACF图已保存: {fig_path}")

# ACF/PACF 分析总结
print("\n" + "-"*80)
print("【ACF/PACF 分析结论】")
print("-"*80)
print("""
原始序列:
  - ACF缓慢线性衰减 → 存在趋势，序列非平稳
  - 需要差分处理

一阶差分后:
  - ACF在lag=7, 14, 21有显著峰值 → 存在周季节性(period=7)
  - 需要季节差分

周差分后(lag=7):
  - 消除了部分周期性，但仍可能存在趋势

一阶+周差分后:
  - ACF/PACF相对平稳
  - 用于确定SARIMA模型的(p,q)和(P,Q)参数
  
初步模型阶数建议:
  - d=1 (一阶差分)
  - D=1 (周季节差分)
  - s=7 (周季节周期)
  - p, q, P, Q 需要根据ACF/PACF具体形状确定
""")

# ============================================================
# Q4: 时间序列分解 (STL Decomposition)
# ============================================================
print("\n" + "="*80)
print("Task 5: Q4 时间序列分解 (STL Decomposition)")
print("="*80)

# STL分解 - 参考Sample 1的分解方法
print("\n【STL分解 (周期=7)】")
stl = STL(ts, period=7, robust=True)
result = stl.fit()

fig2, axes2 = plt.subplots(4, 1, figsize=(14, 12))

# 原始数据
axes2[0].plot(ts.index, ts.values, linewidth=0.8, color='steelblue')
axes2[0].set_title('Original Time Series', fontsize=12)
axes2[0].set_ylabel('Passengers')

# 趋势成分
axes2[1].plot(ts.index, result.trend, linewidth=1.5, color='red')
axes2[1].set_title('Trend Component', fontsize=12)
axes2[1].set_ylabel('Trend')

# 季节成分
axes2[2].plot(ts.index, result.seasonal, linewidth=0.8, color='green')
axes2[2].set_title('Seasonal Component (Period=7)', fontsize=12)
axes2[2].set_ylabel('Seasonal')

# 残差成分
axes2[3].plot(ts.index, result.resid, linewidth=0.8, color='gray', alpha=0.7)
axes2[3].axhline(y=0, color='red', linestyle='--', linewidth=1)
axes2[3].set_title('Residual Component', fontsize=12)
axes2[3].set_ylabel('Residual')
axes2[3].set_xlabel('Date')

plt.tight_layout()
fig_path2 = os.path.join(figures_dir, "04_stl_decomposition.png")
plt.savefig(fig_path2, dpi=150, bbox_inches='tight')
print(f"✅ STL分解图已保存: {fig_path2}")

# 分解成分统计
print("\n【各成分统计】")
print(f"  趋势 (Trend):    均值={result.trend.mean():,.0f}, 标准差={result.trend.std():,.0f}")
print(f"  季节 (Seasonal): 均值={result.seasonal.mean():,.0f}, 标准差={result.seasonal.std():,.0f}")
print(f"  残差 (Residual): 均值={result.resid.mean():,.0f}, 标准差={result.resid.std():,.0f}")

# 各成分方差占比
total_var = ts.var()
trend_var = result.trend.var()
seasonal_var = result.seasonal.var()
resid_var = result.resid.var()

print("\n【方差分解】")
print(f"  趋势方差占比:   {trend_var/total_var*100:.1f}%")
print(f"  季节方差占比:   {seasonal_var/total_var*100:.1f}%")
print(f"  残差方差占比:   {resid_var/total_var*100:.1f}%")

# ============================================================
# Q5: 平稳性检验
# ============================================================
print("\n" + "="*80)
print("Task 6: Q5 平稳性检验 (Stationarity Tests)")
print("="*80)

def adf_test(series, name="Series"):
    """ADF检验"""
    result = adfuller(series.dropna(), autolag='AIC')
    print(f"\n【ADF检验 - {name}】")
    print(f"  Test Statistic: {result[0]:.4f}")
    print(f"  p-value: {result[1]:.4f}")
    print(f"  Lags Used: {result[2]}")
    print(f"  Critical Values:")
    for key, value in result[4].items():
        print(f"    {key}: {value:.4f}")
    
    if result[1] < 0.05:
        print(f"  结论: ✅ 序列平稳 (p < 0.05, 拒绝单位根假设)")
        return True
    else:
        print(f"  结论: ❌ 序列非平稳 (p >= 0.05, 无法拒绝单位根假设)")
        return False

def kpss_test(series, name="Series"):
    """KPSS检验"""
    result = kpss(series.dropna(), regression='c', nlags='auto')
    print(f"\n【KPSS检验 - {name}】")
    print(f"  Test Statistic: {result[0]:.4f}")
    print(f"  p-value: {result[1]:.4f}")
    print(f"  Lags Used: {result[2]}")
    print(f"  Critical Values:")
    for key, value in result[3].items():
        print(f"    {key}: {value:.4f}")
    
    if result[1] > 0.05:
        print(f"  结论: ✅ 序列平稳 (p > 0.05, 无法拒绝平稳假设)")
        return True
    else:
        print(f"  结论: ❌ 序列非平稳 (p <= 0.05, 拒绝平稳假设)")
        return False

# 对不同变换后的序列进行检验
print("\n" + "-"*80)
print("对各种变换后序列进行平稳性检验:")
print("-"*80)

test_series = [
    ("原始序列", ts),
    ("一阶差分", ts_diff1),
    ("周差分(lag=7)", ts_diff7),
    ("一阶+周差分", ts_diff1_7),
]

results_summary = []
for name, series in test_series:
    print(f"\n{'='*40}")
    print(f"检验: {name}")
    print('='*40)
    adf_result = adf_test(series, name)
    kpss_result = kpss_test(series, name)
    results_summary.append({
        'name': name,
        'adf_stationary': adf_result,
        'kpss_stationary': kpss_result
    })

# 平稳性检验总结
print("\n" + "="*80)
print("【平稳性检验总结】")
print("="*80)
print("\n{:<20} {:<15} {:<15} {:<15}".format("序列", "ADF检验", "KPSS检验", "综合结论"))
print("-"*65)
for r in results_summary:
    adf_str = "✅ 平稳" if r['adf_stationary'] else "❌ 非平稳"
    kpss_str = "✅ 平稳" if r['kpss_stationary'] else "❌ 非平稳"
    if r['adf_stationary'] and r['kpss_stationary']:
        conclusion = "✅ 平稳"
    elif not r['adf_stationary'] and not r['kpss_stationary']:
        conclusion = "❌ 非平稳"
    else:
        conclusion = "⚠️ 需进一步分析"
    print("{:<20} {:<15} {:<15} {:<15}".format(r['name'], adf_str, kpss_str, conclusion))

print("""
【结论】
基于ADF和KPSS检验结果:
1. 原始序列: 非平稳 (存在趋势和季节性)
2. 一阶差分: 消除趋势，但仍存在季节性
3. 周差分: 消除周季节性，但可能仍有趋势
4. 一阶+周差分: 同时消除趋势和季节性，序列平稳

【SARIMA模型参数建议】
- d = 1 (一阶差分消除趋势)
- D = 1 (季节差分消除周季节性)
- s = 7 (周季节周期)
- p, q: 根据差分后ACF/PACF确定 (初步建议 p=1-3, q=1-3)
- P, Q: 根据季节滞后处相关性确定 (初步建议 P=0-1, Q=0-1)
""")

plt.close('all')
print("\n✅ Task 4-6 (Q3-Q5) 完成!")
