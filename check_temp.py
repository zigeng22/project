import requests
import pandas as pd
from io import StringIO

url = 'https://data.weather.gov.hk/weatherAPI/cis/csvfile/HKO/ALL/daily_HKO_TEMP_ALL.csv'
response = requests.get(url, timeout=30)
df = pd.read_csv(StringIO(response.text), skiprows=2, encoding='utf-8')
df.columns = ['Year', 'Month', 'Day', 'Value', 'Completeness']

# 转换为数值
df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df['Month'] = pd.to_numeric(df['Month'], errors='coerce')
df['Day'] = pd.to_numeric(df['Day'], errors='coerce')

# 删除无效行
df = df.dropna(subset=['Year', 'Month', 'Day'])
df = df[df['Value'].notna()]  # 只保留有值的行

df['Year'] = df['Year'].astype(int)
df['Month'] = df['Month'].astype(int)
df['Day'] = df['Day'].astype(int)

# 使用errors='coerce'处理无效日期
df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']], errors='coerce')
df = df[df['Date'].notna()]  # 删除无效日期行

print('='*80)
print('Daily Mean Temperature (HK Observatory)')
print('='*80)
print(f'Total rows: {len(df):,}')

date_min = df['Date'].min()
date_max = df['Date'].max()
print(f'Date range: {date_min.strftime("%Y-%m-%d")} to {date_max.strftime("%Y-%m-%d")}')
print(f'Years span: {(date_max - date_min).days / 365.25:.1f} years')
print(f'Completeness: {(df["Completeness"] == "C").mean() * 100:.1f}%')
print(f'Statistics: min={df["Value"].min():.1f}, max={df["Value"].max():.1f}, mean={df["Value"].mean():.1f}')
print()
print('Sample data (first 5 rows):')
print(df[['Date', 'Value']].head().to_string(index=False))
print()
print('Sample data (last 5 rows):')
print(df[['Date', 'Value']].tail().to_string(index=False))
