"""
探索香港政府数据集，寻找适合时间序列分析的数据
"""
import requests
import json
from datetime import datetime

BASE_URL = "https://app.data.gov.hk/v1/historical-archive/list-files"

# 适合时间序列分析的类别
GOOD_CATEGORIES = [
    ("climate-and-weather", "气象"),
    ("transport", "运输"),
    ("finance", "财经"),
    ("environment", "环境"),
    ("health", "卫生"),
    ("population", "人口"),
    ("commerce-and-industry", "工商业"),
    ("housing", "房屋"),
]

# 适合时间序列的数据提供者
GOOD_PROVIDERS = [
    ("hk-hko", "香港天文台"),
    ("hk-td", "运输署"),
    ("hk-censtatd", "政府统计处"),
    ("hk-epd", "环境保护署"),
    ("hk-md", "海事处"),
    ("mtr", "香港铁路有限公司"),
    ("hk-hkma", "香港金融管理局"),
]

def query_api(start, end, category=None, provider=None, format=None, search=None, max_results=100):
    """调用香港政府数据API"""
    params = {
        "start": start,
        "end": end,
        "max": max_results,
    }
    if category:
        params["category"] = category
    if provider:
        params["provider"] = provider
    if format:
        params["format"] = format
    if search:
        params["search"] = search
    
    try:
        response = requests.get(BASE_URL, params=params, timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.status_code}")
            return None
    except Exception as e:
        print(f"Request error: {e}")
        return None

def explore_categories():
    """探索各个类别的数据"""
    print("=" * 80)
    print("探索香港政府数据集 - 寻找适合时间序列分析的数据")
    print("=" * 80)
    
    # 设置时间范围（最近5年的数据）
    start_date = "20190101"
    end_date = "20241231"
    
    all_results = []
    
    # 按类别探索
    for cat_id, cat_name in GOOD_CATEGORIES:
        print(f"\n--- 类别: {cat_name} ({cat_id}) ---")
        result = query_api(start_date, end_date, category=cat_id, max_results=50)
        
        if result and "files" in result:
            files = result["files"]
            total = result.get("resultCount", len(files))
            print(f"  找到 {total} 个文件")
            
            # 统计数据集
            datasets = {}
            for f in files:
                ds_name = f.get("dataset-tc", f.get("dataset-en", "Unknown"))
                provider = f.get("provider", "Unknown")
                if ds_name not in datasets:
                    datasets[ds_name] = {
                        "provider": provider,
                        "count": 0,
                        "formats": set(),
                        "sample_url": f.get("url", "")
                    }
                datasets[ds_name]["count"] += 1
                datasets[ds_name]["formats"].add(f.get("url", "").split(".")[-1])
            
            # 显示数据集（按文件数量排序）
            sorted_datasets = sorted(datasets.items(), key=lambda x: x[1]["count"], reverse=True)
            for ds_name, info in sorted_datasets[:5]:  # 只显示前5个
                print(f"    • {ds_name}")
                print(f"      提供者: {info['provider']}, 文件数: {info['count']}, 格式: {info['formats']}")
            
            all_results.extend([(cat_name, ds_name, info) for ds_name, info in datasets.items()])
    
    return all_results

def explore_weather_data():
    """专门探索气象数据（最适合时间序列）"""
    print("\n" + "=" * 80)
    print("重点探索: 香港天文台气象数据")
    print("=" * 80)
    
    # 查询天文台数据
    result = query_api("20150101", "20241231", provider="hk-hko", max_results=200)
    
    if result and "files" in result:
        files = result["files"]
        print(f"找到 {result.get('resultCount', len(files))} 个天文台数据文件")
        
        # 按数据集分组
        datasets = {}
        for f in files:
            ds_name = f.get("dataset-tc", f.get("dataset-en", "Unknown"))
            resource = f.get("resource-tc", f.get("resource-en", ""))
            url = f.get("url", "")
            
            if ds_name not in datasets:
                datasets[ds_name] = []
            datasets[ds_name].append({
                "resource": resource,
                "url": url,
                "format": url.split(".")[-1] if "." in url else "unknown"
            })
        
        print(f"\n共有 {len(datasets)} 个不同的数据集:")
        for ds_name, files in sorted(datasets.items(), key=lambda x: len(x[1]), reverse=True):
            print(f"\n📊 {ds_name}")
            print(f"   文件数量: {len(files)}")
            formats = set(f["format"] for f in files)
            print(f"   格式: {formats}")
            # 显示前3个资源
            for f in files[:3]:
                print(f"   - {f['resource'][:50]}..." if len(f['resource']) > 50 else f"   - {f['resource']}")

def explore_transport_data():
    """探索运输数据"""
    print("\n" + "=" * 80)
    print("重点探索: 运输数据")
    print("=" * 80)
    
    result = query_api("20150101", "20241231", category="transport", max_results=200)
    
    if result and "files" in result:
        files = result["files"]
        print(f"找到 {result.get('resultCount', len(files))} 个运输数据文件")
        
        datasets = {}
        for f in files:
            ds_name = f.get("dataset-tc", f.get("dataset-en", "Unknown"))
            provider = f.get("provider", "Unknown")
            
            if ds_name not in datasets:
                datasets[ds_name] = {"provider": provider, "count": 0, "files": []}
            datasets[ds_name]["count"] += 1
            datasets[ds_name]["files"].append(f)
        
        print(f"\n共有 {len(datasets)} 个不同的数据集:")
        for ds_name, info in sorted(datasets.items(), key=lambda x: x[1]["count"], reverse=True)[:10]:
            print(f"\n📊 {ds_name}")
            print(f"   提供者: {info['provider']}, 文件数: {info['count']}")

def explore_specific_search():
    """搜索特定关键词的数据"""
    print("\n" + "=" * 80)
    print("关键词搜索: 时间序列相关数据")
    print("=" * 80)
    
    keywords = ["daily", "monthly", "temperature", "passenger", "traffic", "pollution", "rainfall"]
    
    for keyword in keywords:
        result = query_api("20190101", "20241231", search=keyword, max_results=20)
        if result and "files" in result:
            count = result.get("resultCount", 0)
            print(f"\n🔍 关键词 '{keyword}': 找到 {count} 个结果")
            
            if count > 0:
                # 显示部分结果
                for f in result["files"][:3]:
                    ds_name = f.get("dataset-tc", f.get("dataset-en", "Unknown"))
                    print(f"   - {ds_name}")

if __name__ == "__main__":
    print("开始探索香港政府数据集...\n")
    
    # 1. 探索各类别
    explore_categories()
    
    # 2. 重点探索气象数据
    explore_weather_data()
    
    # 3. 探索运输数据
    explore_transport_data()
    
    # 4. 关键词搜索
    explore_specific_search()
    
    print("\n" + "=" * 80)
    print("探索完成！")
    print("=" * 80)
