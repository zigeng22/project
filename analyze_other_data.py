import requests
import pandas as pd
from io import StringIO

print("="*90)
print("香港非天气类时间序列数据集详细分析")
print("="*90)

datasets_to_check = [
    ('入境处-每日过境旅客统计', 'https://www.immd.gov.hk/opendata/eng/transport/immigration_clearance/statistics_on_daily_passenger_traffic.csv'),
    ('机电署-电力统计', 'https://data.gov.hk/tc-data/dataset/hk-emsd-emsd1-electricity-consumption-hongkong'),
]

# 1. 入境处数据
print("\n" + "="*90)
print("🛂 入境处 - 每日过境旅客统计")
print("="*90)
url = 'https://www.immd.gov.hk/opendata/eng/transport/immigration_clearance/statistics_on_daily_passenger_traffic.csv'
response = requests.get(url, timeout=30)
df = pd.read_csv(StringIO(response.text))

# 清理列名
df.columns = ['Date', 'Control_Point', 'Direction', 'HK_Residents', 'Mainland_Visitors', 
              'Other_Visitors', 'Total', 'Control_Point_CN']

# 转换日期
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')

print(f"总行数: {len(df):,}")
print(f"日期范围: {df['Date'].min().strftime('%Y-%m-%d')} 到 {df['Date'].max().strftime('%Y-%m-%d')}")
print(f"时间跨度: {(df['Date'].max() - df['Date'].min()).days / 365.25:.1f} 年")
print(f"\n口岸列表:")
for cp in df['Control_Point'].unique():
    print(f"  - {cp}")

print(f"\n方向: {df['Direction'].unique().tolist()}")

# 按日期汇总总客流
daily_total = df.groupby('Date')['Total'].sum().reset_index()
print(f"\n按日汇总后行数: {len(daily_total):,}")
print(f"每日总客流统计: min={daily_total['Total'].min():,.0f}, max={daily_total['Total'].max():,.0f}, mean={daily_total['Total'].mean():,.0f}")

# 可以按口岸分析
print("\n各口岸数据量:")
for cp in df['Control_Point'].unique()[:5]:
    cp_data = df[df['Control_Point'] == cp]
    cp_daily = cp_data.groupby('Date')['Total'].sum()
    print(f"  {cp}: {len(cp_daily)} 天数据")

# 2. 探索data.gov.hk的其他数据集
print("\n" + "="*90)
print("🔍 探索 data.gov.hk 其他开放数据")
print("="*90)

other_urls = [
    ('股票每日成交', 'https://www.hkex.com.hk/-/media/HKEX-Market/Market-Data/Statistics/Consolidated-Reports/Annual-Market-Statistics/2023-statistics.xlsx'),
    ('空气质量-一般监测站', 'https://cd.epic.epd.gov.hk/EPICDI/air/download/?lang=en'),
]

# 3. 环保署空气质量数据
print("\n" + "="*90) 
print("🌫️ 环保署 - 空气质量数据")
print("="*90)
# 空气质量数据URL格式
aqhi_url = 'https://www.aqhi.gov.hk/epd/ddata/html/out/24aqhi_Eng.csv'
try:
    response = requests.get(aqhi_url, timeout=30)
    if response.status_code == 200:
        print(f"空气质量实时数据可用")
        print(f"内容预览:\n{response.text[:500]}")
except Exception as e:
    print(f"Error: {e}")

# 4. 天文台的其他有趣数据
print("\n" + "="*90)
print("🌊 天文台 - 潮汐数据 (非温度)")
print("="*90)
tide_url = 'https://data.weather.gov.hk/weatherAPI/opendata/opendata.php?dataType=HHOT&lang=en&rformat=csv'
try:
    response = requests.get(tide_url, timeout=30)
    if response.status_code == 200:
        print(f"潮汐数据可用")
        lines = response.text.split('\n')[:10]
        for line in lines:
            print(f"  {line}")
except Exception as e:
    print(f"Error: {e}")
