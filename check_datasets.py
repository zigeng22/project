import requests
import pandas as pd
from io import StringIO

datasets_to_check = [
    ('Daily Mean Relative Humidity (Sheung Shui)', 'https://data.weather.gov.hk/weatherAPI/cis/csvfile/SSH/ALL/daily_SSH_RH_ALL.csv'),
    ('Daily Mean Dew Point (Sheung Shui)', 'https://data.weather.gov.hk/weatherAPI/cis/csvfile/SSH/ALL/daily_SSH_DEW_ALL.csv'),
    ('Daily Mean Pressure (Sheung Shui)', 'https://data.weather.gov.hk/weatherAPI/cis/csvfile/SSH/ALL/daily_SSH_MSLP_ALL.csv'),
    ('Daily Total Sunshine (King Park)', 'https://data.weather.gov.hk/weatherAPI/cis/csvfile/KP/ALL/daily_KP_SUN_ALL.csv'),
    ('Daily Total Rainfall (Sai Kung)', 'https://data.weather.gov.hk/weatherAPI/cis/csvfile/SSP/ALL/daily_SSP_RF_ALL.csv'),
    ('Daily Mean UV Index (King Park)', 'https://data.weather.gov.hk/weatherAPI/cis/csvfile/KP/ALL/daily_KP_UV_ALL.csv'),
    ('Daily Mean Temperature (HK Observatory)', 'https://data.weather.gov.hk/weatherAPI/cis/csvfile/HKO/ALL/daily_HKO_TEMP_ALL.csv'),
    ('Daily Max Temperature (HK Observatory)', 'https://data.weather.gov.hk/weatherAPI/cis/csvfile/HKO/ALL/daily_HKO_MAXT_ALL.csv'),
]

print('='*90)
print('Hong Kong Observatory Dataset Quality Analysis')
print('='*90)

results = []
for name, url in datasets_to_check:
    print(f'\n{name}')
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text), skiprows=2, encoding='utf-8')
            df.columns = ['Year', 'Month', 'Day', 'Value', 'Completeness']
            
            # 处理缺失值 - 转换为数值，错误值变为NaN
            df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
            df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
            df['Month'] = pd.to_numeric(df['Month'], errors='coerce')
            df['Day'] = pd.to_numeric(df['Day'], errors='coerce')
            
            # 删除年月日有缺失的行
            df = df.dropna(subset=['Year', 'Month', 'Day'])
            df['Year'] = df['Year'].astype(int)
            df['Month'] = df['Month'].astype(int)
            df['Day'] = df['Day'].astype(int)
            
            df['Date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
            
            n_rows = len(df)
            date_start = df['Date'].min().strftime('%Y-%m-%d')
            date_end = df['Date'].max().strftime('%Y-%m-%d')
            years_span = (df['Date'].max() - df['Date'].min()).days / 365.25
            complete_pct = (df['Completeness'] == 'C').mean() * 100
            missing_value_pct = df['Value'].isna().mean() * 100
            
            # 只统计非缺失值
            valid_values = df['Value'].dropna()
            if len(valid_values) > 0:
                val_min = valid_values.min()
                val_max = valid_values.max()
                val_mean = valid_values.mean()
            else:
                val_min = val_max = val_mean = 0
            
            print(f'  Rows: {n_rows:,} | Date: {date_start} to {date_end} ({years_span:.1f} years)')
            print(f'  Complete flag: {complete_pct:.1f}% | Missing values: {missing_value_pct:.1f}%')
            print(f'  Stats: min={val_min:.1f}, max={val_max:.1f}, mean={val_mean:.1f}')
            
            results.append({
                'name': name, 
                'rows': n_rows, 
                'years': years_span, 
                'complete': complete_pct, 
                'url': url,
                'start': date_start,
                'end': date_end
            })
    except Exception as e:
        print(f'  Error: {e}')

print('\n' + '='*90)
print('RANKING by data quantity:')
print('='*90)
results.sort(key=lambda x: x['rows'], reverse=True)
for i, r in enumerate(results, 1):
    name = r['name']
    rows = r['rows']
    years = r['years']
    complete = r['complete']
    print(f'{i}. {name}')
    print(f'   {rows:,} rows | {years:.1f} years | {complete:.1f}% complete')

print('\n' + '='*90)
print('FINAL RECOMMENDATION:')
print('='*90)
best = results[0]
print(f'''
Best Dataset: {best['name']}
- Data URL: {best['url']}
- Sample Size: {best['rows']:,} (n >> 100, excellent!)
- Time Span: {best['start']} to {best['end']} ({best['years']:.1f} years)
- Data Completeness: {best['complete']:.1f}%

Why this dataset is ideal for time series project:
1. Large sample size ({best['rows']:,} daily observations)
2. Long time span ({best['years']:.1f} years) - good for capturing seasonality
3. High data completeness ({best['complete']:.1f}%)
4. Daily frequency - suitable for time series analysis
5. Weather data typically shows clear seasonal patterns
''')
