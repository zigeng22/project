import requests
import pandas as pd
from io import StringIO
from datetime import datetime

BASE_URL = "https://app.data.gov.hk/v1/historical-archive/list-files"

def search_datasets(provider=None, search=None, start="20150101", end="20251231"):
    """搜索数据集"""
    params = {
        "start": start,
        "end": end,
        "format": "csv",
        "max": 500
    }
    if provider:
        params["provider"] = provider
    if search:
        params["search"] = search
    
    response = requests.get(BASE_URL, params=params)
    if response.status_code == 200:
        return response.json()
    return None

def analyze_dataset(url, name):
    """分析数据集质量"""
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            # 尝试不同的解析方式
            text = response.text
            lines = text.split('\n')
            
            # 找到数据开始的行（跳过标题行）
            skip_rows = 0
            for i, line in enumerate(lines[:5]):
                if ',' in line and not any(c.isalpha() for c in line.split(',')[0] if c not in ['/', '-']):
                    break
                skip_rows = i + 1
            
            df = pd.read_csv(StringIO(text), skiprows=skip_rows, encoding='utf-8', on_bad_lines='skip')
            return {
                'name': name,
                'rows': len(df),
                'cols': len(df.columns),
                'columns': list(df.columns)[:5],
                'url': url
            }
    except Exception as e:
        return {'name': name, 'error': str(e)[:50], 'url': url}

print("="*90)
print("探索香港政府开放数据 - 交通及其他领域数据集 (2015-2025)")
print("="*90)

# 1. 交通署数据 (hk-td)
print("\n" + "="*90)
print("🚗 交通署 (hk-td) 数据集")
print("="*90)
data = search_datasets(provider="hk-td")
if data:
    datasets = {}
    for item in data:
        name = item.get('dataset-name-en', item.get('dataset-name-tc', 'Unknown'))
        if name not in datasets:
            datasets[name] = []
        datasets[name].append(item)
    
    print(f"找到 {len(datasets)} 个不同数据集类型")
    for name, items in sorted(datasets.items(), key=lambda x: -len(x[1]))[:15]:
        print(f"  - {name}: {len(items)} 个文件")
        if items:
            print(f"    URL示例: {items[0].get('url-link', 'N/A')[:80]}")

# 2. MTR数据
print("\n" + "="*90)
print("🚇 港铁 (mtr) 数据集")
print("="*90)
data = search_datasets(provider="mtr")
if data:
    datasets = {}
    for item in data:
        name = item.get('dataset-name-en', item.get('dataset-name-tc', 'Unknown'))
        if name not in datasets:
            datasets[name] = []
        datasets[name].append(item)
    
    print(f"找到 {len(datasets)} 个不同数据集类型")
    for name, items in sorted(datasets.items(), key=lambda x: -len(x[1]))[:10]:
        print(f"  - {name}: {len(items)} 个文件")
        if items:
            print(f"    URL示例: {items[0].get('url-link', 'N/A')[:80]}")

# 3. 统计处数据 (hk-censtatd) - 经济统计
print("\n" + "="*90)
print("📊 统计处 (hk-censtatd) 数据集")
print("="*90)
data = search_datasets(provider="hk-censtatd")
if data:
    datasets = {}
    for item in data:
        name = item.get('dataset-name-en', item.get('dataset-name-tc', 'Unknown'))
        if name not in datasets:
            datasets[name] = []
        datasets[name].append(item)
    
    print(f"找到 {len(datasets)} 个不同数据集类型")
    for name, items in sorted(datasets.items(), key=lambda x: -len(x[1]))[:15]:
        print(f"  - {name}: {len(items)} 个文件")

# 4. 搜索特定关键词
keywords = ["traffic", "passenger", "vehicle", "electricity", "water", "retail", "tourism"]
print("\n" + "="*90)
print("🔍 关键词搜索结果")
print("="*90)
for kw in keywords:
    data = search_datasets(search=kw)
    if data:
        print(f"\n'{kw}': 找到 {len(data)} 个文件")
        # 显示前3个
        seen = set()
        for item in data[:10]:
            name = item.get('dataset-name-en', '')
            if name and name not in seen:
                seen.add(name)
                print(f"  - {name}")
