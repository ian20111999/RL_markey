"""
Complete Test Runner
執行所有測試並生成報告
"""
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import json

project_root = Path(__file__).parent


def run_command(cmd, description):
    """運行命令並返回結果"""
    print(f"\n{'='*80}")
    print(f"🧪 {description}")
    print(f"{'='*80}\n")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=project_root
        )
        
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    print("\n" + "="*80)
    print("🧪 RL Market - Complete Test Suite")
    print("="*80 + "\n")
    
    # Check if pytest is installed
    try:
        import pytest
        print("✅ pytest found")
    except ImportError:
        print("❌ pytest not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pytest", "pytest-cov"])
    
    # Create logs directory
    (project_root / "logs").mkdir(exist_ok=True)
    
    results = {}
    
    # Test suites
    test_suites = [
        {
            'name': 'Environment Tests',
            'file': 'tests/test_env_basic.py',
            'description': 'Testing trading environment'
        },
        {
            'name': 'Reward Tests',
            'file': 'tests/test_reward.py',
            'description': 'Testing reward functions'
        },
        {
            'name': 'Database Tests',
            'file': 'tests/test_database.py',
            'description': 'Testing database operations'
        },
        {
            'name': 'Production Tests',
            'file': 'tests/test_production.py',
            'description': 'Testing production module'
        },
        {
            'name': 'Integration Tests',
            'file': 'tests/test_integration.py',
            'description': 'Testing system integration'
        },
        {
            'name': 'API Tests',
            'file': 'tests/test_api.py',
            'description': 'Testing API endpoints'
        }
    ]
    
    # Run each test suite
    for i, suite in enumerate(test_suites, 1):
        test_file = project_root / suite['file']
        
        if not test_file.exists():
            print(f"⚠️  {suite['name']}: File not found - {suite['file']}")
            results[suite['name']] = 'skipped'
            continue
        
        print(f"\n{i}️⃣  {suite['description']}...")
        
        success = run_command(
            [sys.executable, "-m", "pytest", str(test_file), "-v", "--tb=short"],
            suite['name']
        )
        
        results[suite['name']] = 'passed' if success else 'failed'
    
    # Print summary
    print("\n" + "="*80)
    print("📊 Test Summary")
    print("="*80 + "\n")
    
    passed = sum(1 for r in results.values() if r == 'passed')
    failed = sum(1 for r in results.values() if r == 'failed')
    skipped = sum(1 for r in results.values() if r == 'skipped')
    
    for name, status in results.items():
        icon = "✅" if status == "passed" else "❌" if status == "failed" else "⚠️"
        print(f"{icon} {name}: {status.upper()}")
    
    print(f"\n{'='*80}")
    print(f"Total: {len(results)} | Passed: {passed} | Failed: {failed} | Skipped: {skipped}")
    print(f"{'='*80}\n")
    
    # Save results
    report = {
        'timestamp': datetime.now().isoformat(),
        'results': results,
        'summary': {
            'total': len(results),
            'passed': passed,
            'failed': failed,
            'skipped': skipped
        }
    }
    
    report_path = project_root / 'logs' / 'test_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"📄 Test report saved to: {report_path}")
    
    # Ask for coverage
    if failed == 0:
        print("\n🎉 All tests passed!")
        response = input("\nGenerate coverage report? (y/n): ").strip().lower()
        
        if response == 'y':
            print("\n📊 Generating coverage report...")
            subprocess.run([
                sys.executable, "-m", "pytest",
                "tests/",
                "--cov=.",
                "--cov-report=html",
                "--cov-report=term"
            ], cwd=project_root)
            print(f"\n📄 Coverage report: {project_root}/htmlcov/index.html")
    else:
        print(f"\n⚠️  {failed} test suite(s) failed. Fix issues before generating coverage.")
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
