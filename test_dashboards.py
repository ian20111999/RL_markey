#!/usr/bin/env python3
"""
測試所有儀表板功能
"""
import sys
import time
import subprocess
import requests
from pathlib import Path

def test_dashboard(name, script, port, expected_endpoints):
    """測試單個儀表板"""
    print(f"\n{'='*60}")
    print(f"測試 {name}")
    print(f"{'='*60}")
    
    # 啟動服務
    print(f"⏳ 啟動 {script} 在端口 {port}...")
    process = subprocess.Popen(
        [sys.executable, script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=Path(__file__).parent
    )
    
    # 等待服務啟動
    max_wait = 10
    base_url = f"http://localhost:{port}"
    
    for i in range(max_wait):
        time.sleep(1)
        try:
            response = requests.get(f"{base_url}/api/health", timeout=2)
            if response.status_code == 200:
                print(f"✅ 服務已啟動 (等待 {i+1} 秒)")
                break
        except requests.exceptions.RequestException:
            if i == max_wait - 1:
                print(f"❌ 服務啟動超時")
                process.kill()
                return False
            continue
    
    # 測試各個 endpoint
    results = []
    for endpoint in expected_endpoints:
        url = f"{base_url}{endpoint}"
        try:
            response = requests.get(url, timeout=5)
            status = "✅" if response.status_code == 200 else "❌"
            results.append((endpoint, response.status_code, status))
            print(f"{status} {endpoint} - 狀態碼: {response.status_code}")
            
            # 如果是 JSON，顯示部分內容
            if response.headers.get('Content-Type', '').startswith('application/json'):
                data = response.json()
                if isinstance(data, dict):
                    keys = list(data.keys())[:5]
                    print(f"   返回資料鍵: {keys}")
        except Exception as e:
            results.append((endpoint, None, "❌"))
            print(f"❌ {endpoint} - 錯誤: {str(e)[:50]}")
    
    # 停止服務
    process.kill()
    process.wait()
    print(f"\n🛑 已停止 {name}")
    
    # 總結
    success = all(r[2] == "✅" for r in results)
    return success


def main():
    print("🧪 RL Market Making - 儀表板功能測試")
    print("="*60)
    
    # 檢查資料庫
    db_path = Path(__file__).parent / "logs" / "metrics.db"
    if not db_path.exists():
        print(f"⚠️  警告: 找不到 metrics.db")
        print(f"   路徑: {db_path}")
        print("   某些功能可能無法測試")
    else:
        print(f"✅ 找到資料庫: {db_path} ({db_path.stat().st_size / 1024:.1f} KB)")
    
    # 測試配置
    tests = [
        {
            "name": "基礎儀表板 (web_dashboard.py)",
            "script": "web_dashboard.py",
            "port": 5555,
            "endpoints": [
                "/",
                "/api/health",
                "/api/dashboard",
                "/api/stats"
            ]
        },
        {
            "name": "增強版儀表板 (monitoring_dashboard.py)",
            "script": "monitoring_dashboard.py", 
            "port": 5556,
            "endpoints": [
                "/",
                "/api/health",
                "/api/dashboard"
            ]
        }
    ]
    
    results = {}
    for test_config in tests:
        try:
            success = test_dashboard(
                test_config["name"],
                test_config["script"],
                test_config["port"],
                test_config["endpoints"]
            )
            results[test_config["name"]] = success
        except KeyboardInterrupt:
            print("\n\n⚠️  測試被使用者中斷")
            sys.exit(1)
        except Exception as e:
            print(f"\n❌ 測試失敗: {e}")
            results[test_config["name"]] = False
    
    # 最終報告
    print("\n" + "="*60)
    print("📊 測試總結")
    print("="*60)
    for name, success in results.items():
        status = "✅ 通過" if success else "❌ 失敗"
        print(f"{status} - {name}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 所有測試通過！")
        return 0
    else:
        print("\n⚠️  部分測試失敗")
        return 1


if __name__ == "__main__":
    sys.exit(main())
