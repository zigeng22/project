import requests

url = 'https://app.data.gov.hk/v1/historical-archive/list-files'

# 搜索不同类型的数据
searches = [
    ('traffic', '交通流量'),
    ('passenger', '乘客量'),
    ('vehicle', '车辆'),
    ('border', '过境'),
    ('retail', '零售'),
    ('tourism', '旅游'),
    ('electricity', '电力'),
    ('air quality', '空气质量'),
    ('stock', '股票'),
    ('property', '房产'),
]

print("="*90)
print("搜索香港政府开放数据 - 非天气类数据集")
print("="*90)

all_results = []

for search_en, search_cn in searches:
    params = {'start': '20150101', 'end': '20251231', 'search': search_en, 'max': 100}
    response = requests.get(url, params=params)
    data = response.json()
    
    if isinstance(data, list) and len(data) > 0:
        print(f"\n🔍 '{search_en}' ({search_cn}): {len(data)} 个文件")
        
        # 按数据集名称分组
        datasets = {}
        for item in data:
            name = item.get('dataset-name-en', item.get('dataset-name-tc', 'Unknown'))
            if name not in datasets:
                datasets[name] = {'count': 0, 'url': item.get('url-link', '')}
            datasets[name]['count'] += 1
        
        for name, info in sorted(datasets.items(), key=lambda x: -x['count'])[:5]:
            print(f"  [{info['count']} files] {name}")
            all_results.append({
                'category': search_en,
                'name': name,
                'count': info['count'],
                'url': info['url']
            })

# 直接搜索已知的一些数据集URL
print("\n" + "="*90)
print("直接检查已知的交通/经济数据API")
print("="*90)

known_apis = [
    ('港铁乘客量', 'https://opendata.mtr.com.hk/data/passenger_traffic_data.csv'),
    ('过境旅客统计', 'https://www.immd.gov.hk/opendata/eng/transport/immigration_clearance/statistics_on_daily_passenger_traffic.csv'),
]

import pandas as pd
from io import StringIO

for name, url in known_apis:
    print(f"\n📊 {name}")
    print(f"   URL: {url}")
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            df = pd.read_csv(StringIO(response.text), on_bad_lines='skip')
            print(f"   ✅ 行数: {len(df)}, 列数: {len(df.columns)}")
            print(f"   列名: {list(df.columns)[:5]}")
            if len(df) > 0:
                print(f"   首行: {df.iloc[0].tolist()[:5]}")
        else:
            print(f"   ❌ HTTP {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
