import requests
import pandas as pd
from io import StringIO

print("="*90)
print("🚌 香港交通/出行相关时间序列数据集 - 详细质量分析")
print("="*90)

# ==========================================
# 1. 入境处过境旅客数据 - 详细分析
# ==========================================
print("\n" + "-"*90)
print("📊 数据集1: 入境处每日过境旅客统计")
print("-"*90)

url = 'https://www.immd.gov.hk/opendata/eng/transport/immigration_clearance/statistics_on_daily_passenger_traffic.csv'
response = requests.get(url, timeout=30)
df = pd.read_csv(StringIO(response.text))
df.columns = ['Date', 'Control_Point', 'Direction', 'HK_Residents', 'Mainland_Visitors', 
              'Other_Visitors', 'Total', 'Control_Point_CN']
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

# 分析不同维度的数据
print("\n✨ 可以做的时间序列分析维度:")

# 维度1: 全港每日总客流
daily_total = df.groupby('Date')['Total'].sum().reset_index()
print(f"\n1️⃣ 全港每日总客流")
print(f"   样本数: {len(daily_total)} 天")
print(f"   日期: {daily_total['Date'].min().strftime('%Y-%m-%d')} 至 {daily_total['Date'].max().strftime('%Y-%m-%d')}")
print(f"   统计: min={daily_total['Total'].min():,.0f}, max={daily_total['Total'].max():,.0f}")

# 维度2: 单一口岸
print(f"\n2️⃣ 单一口岸分析 (以机场为例)")
airport = df[df['Control_Point'] == 'Airport'].groupby('Date')['Total'].sum().reset_index()
print(f"   样本数: {len(airport)} 天")
print(f"   统计: min={airport['Total'].min():,.0f}, max={airport['Total'].max():,.0f}")

# 维度3: 按旅客类型
print(f"\n3️⃣ 按旅客类型分析")
for col in ['HK_Residents', 'Mainland_Visitors', 'Other_Visitors']:
    daily = df.groupby('Date')[col].sum()
    print(f"   {col}: mean={daily.mean():,.0f}/天")

# ==========================================
# 2. 检查机场数据
# ==========================================
print("\n" + "-"*90)
print("📊 数据集2: 香港机场统计")
print("-"*90)

airport_urls = [
    'https://www.hongkongairport.com/iwov-resources/file/airport-authority/media/download/statistics/stat-summary-en.pdf',
]

# 机场月度数据 (从data.gov.hk)
print("检查机场相关开放数据...")

# ==========================================
# 3. 检查运输署数据
# ==========================================
print("\n" + "-"*90)
print("📊 数据集3: 运输署交通数据")
print("-"*90)

# 过海隧道流量
td_urls = [
    ('过海隧道交通流量', 'https://data.gov.hk/tc-data/dataset/hk-td-tis_2-traffic-data-through-cross-harbour-driving-route'),
]
print("运输署通常提供月度/年度交通统计，需要从官网下载")

# ==========================================
# 4. 再次检查天文台非温度数据
# ==========================================
print("\n" + "-"*90)
print("📊 数据集4: 天文台其他气象数据 (非温度)")
print("-"*90)

hko_others = [
    ('每日总蒸发量(京士柏)', 'https://data.weather.gov.hk/weatherAPI/cis/csvfile/KP/ALL/daily_KP_EVAP_ALL.csv'),
    ('每日平均云量(京士柏)', 'https://data.weather.gov.hk/weatherAPI/cis/csvfile/KP/ALL/daily_KP_CLD_ALL.csv'),
    ('每日平均能见度(香港国际机场)', 'https://data.weather.gov.hk/weatherAPI/cis/csvfile/HKA/ALL/daily_HKA_VIS_ALL.csv'),
    ('每日总降雨量(天文台)', 'https://data.weather.gov.hk/weatherAPI/cis/csvfile/HKO/ALL/daily_HKO_RF_ALL.csv'),
]

for name, url in hko_others:
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text), skiprows=2)
            df.columns = ['Year', 'Month', 'Day', 'Value', 'Completeness']
            df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
            df = df[df['Value'].notna()]
            df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
            df['Month'] = pd.to_numeric(df['Month'], errors='coerce') 
            df['Day'] = pd.to_numeric(df['Day'], errors='coerce')
            df = df.dropna(subset=['Year', 'Month', 'Day'])
            
            print(f"\n{name}")
            print(f"   样本数: {len(df):,} | 完整性: {(df['Completeness']=='C').mean()*100:.1f}%")
            print(f"   统计: min={df['Value'].min():.1f}, max={df['Value'].max():.1f}, mean={df['Value'].mean():.1f}")
    except Exception as e:
        print(f"\n{name}: Error - {str(e)[:50]}")

# ==========================================
# 总结推荐
# ==========================================
print("\n" + "="*90)
print("🎯 非天气类数据集推荐总结")
print("="*90)
print("""
┌─────────────────────────────────────────────────────────────────────────────────┐
│ 排名 │ 数据集                           │ 样本数  │ 时间跨度 │ 推荐理由        │
├─────────────────────────────────────────────────────────────────────────────────┤
│  1   │ 入境处-每日过境旅客统计          │ 1,812+  │ 5年      │ 数据量大、维度多│
│      │ (可按口岸/方向/旅客类型细分)     │         │ 2021-今  │ COVID恢复趋势   │
├─────────────────────────────────────────────────────────────────────────────────┤
│  2   │ 天文台-每日总降雨量(天文台)      │ 40000+  │ 100+年   │ 历史悠久、完整  │
├─────────────────────────────────────────────────────────────────────────────────┤
│  3   │ 天文台-每日总蒸发量              │ 多年    │ 多年     │ 非温度气象数据  │
└─────────────────────────────────────────────────────────────────────────────────┘
""")
