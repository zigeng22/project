"""
香港政府数据集全面探索 - 寻找适合时间序列分析的数据
"""
import requests
import json
from collections import defaultdict

BASE_URL = "https://app.data.gov.hk/v1/historical-archive/list-files"

def query_api(start, end, **kwargs):
    """调用API"""
    params = {"start": start, "end": end, "max": kwargs.get("max", 500)}
    for key in ["category", "provider", "format", "search"]:
        if key in kwargs and kwargs[key]:
            params[key] = kwargs[key]
    
    try:
        response = requests.get(BASE_URL, params=params, timeout=60)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"Error: {e}")
    return None

def explore_provider(provider_id, provider_name):
    """探索特定数据提供者的数据"""
    print(f"\n{'='*80}")
    print(f"📊 {provider_name} ({provider_id})")
    print('='*80)
    
    result = query_api("20180101", "20241231", provider=provider_id, max=1000)
    
    if not result or "files" not in result:
        print("  无数据")
        return []
    
    files = result["files"]
    total = result.get("file-count", len(files))
    print(f"总共有 {total} 个文件\n")
    
    # 按数据集分组
    datasets = defaultdict(lambda: {
        "name_en": "", "name_tc": "", "files": [], 
        "formats": set(), "has_all_year": False, "sample_url": ""
    })
    
    for f in files:
        ds_id = f.get("dataset-id", "unknown")
        datasets[ds_id]["name_en"] = f.get("dataset-name-en", "")
        datasets[ds_id]["name_tc"] = f.get("dataset-name-tc", "")
        datasets[ds_id]["files"].append(f)
        datasets[ds_id]["formats"].add(f.get("format", ""))
        
        url = f.get("url", "")
        resource = f.get("resource-name-en", "")
        
        # 检查是否有全部年份的数据
        if "ALL" in url or "all year" in resource.lower():
            datasets[ds_id]["has_all_year"] = True
            datasets[ds_id]["sample_url"] = url
        elif not datasets[ds_id]["sample_url"]:
            datasets[ds_id]["sample_url"] = url
    
    # 整理并显示数据集
    dataset_list = []
    for ds_id, info in sorted(datasets.items(), key=lambda x: len(x[1]["files"]), reverse=True):
        dataset_list.append({
            "id": ds_id,
            "name_tc": info["name_tc"],
            "name_en": info["name_en"],
            "file_count": len(info["files"]),
            "formats": list(info["formats"]),
            "has_all_year": info["has_all_year"],
            "sample_url": info["sample_url"]
        })
        
        # 显示信息
        all_year_mark = "✅ 有全部年份数据" if info["has_all_year"] else ""
        print(f"📁 {info['name_tc']} / {info['name_en']}")
        print(f"   ID: {ds_id}")
        print(f"   文件数: {len(info['files'])}, 格式: {info['formats']} {all_year_mark}")
        print(f"   示例URL: {info['sample_url'][:80]}..." if len(info['sample_url']) > 80 else f"   示例URL: {info['sample_url']}")
        print()
    
    return dataset_list

def check_data_sample(url):
    """检查数据样本"""
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            content = response.text
            lines = content.strip().split('\n')
            print(f"  行数: {len(lines)}")
            print(f"  前3行:")
            for line in lines[:3]:
                print(f"    {line[:100]}...")
            return len(lines)
    except Exception as e:
        print(f"  Error: {e}")
    return 0

def main():
    print("=" * 80)
    print("香港政府数据集探索 - 寻找适合时间序列项目的数据")
    print("=" * 80)
    
    # 重点探索的数据提供者
    providers = [
        ("hk-hko", "香港天文台"),
        ("hk-td", "运输署"),
        ("mtr", "香港铁路有限公司"),
        ("hk-epd", "环境保护署"),
        ("hk-censtatd", "政府统计处"),
    ]
    
    all_datasets = {}
    
    for provider_id, provider_name in providers:
        datasets = explore_provider(provider_id, provider_name)
        all_datasets[provider_id] = datasets
    
    # 总结推荐
    print("\n" + "=" * 80)
    print("🌟 推荐数据集总结 (适合时间序列分析)")
    print("=" * 80)
    
    recommendations = []
    
    for provider_id, datasets in all_datasets.items():
        for ds in datasets:
            if ds["has_all_year"] and ds["file_count"] >= 5:
                recommendations.append(ds)
    
    # 按文件数量排序
    recommendations.sort(key=lambda x: x["file_count"], reverse=True)
    
    print("\n以下数据集有'全部年份'数据，适合做时间序列分析:\n")
    for i, ds in enumerate(recommendations[:20], 1):
        print(f"{i}. {ds['name_tc']} / {ds['name_en']}")
        print(f"   文件数: {ds['file_count']}, 格式: {ds['formats']}")
        print(f"   URL: {ds['sample_url']}")
        print()
    
    return recommendations

if __name__ == "__main__":
    recommendations = main()
