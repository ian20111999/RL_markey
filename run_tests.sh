#!/bin/bash
# 完整測試套件執行腳本

set -e  # Exit on error

echo "🧪 RL Market - Complete Test Suite"
echo "=" 
echo ""

# Check if pytest is installed
if ! command -v pytest &> /dev/null; then
    echo "❌ pytest not found. Installing..."
    pip install pytest pytest-cov
fi

# Create logs directory if not exists
mkdir -p logs

echo "📋 Running Test Suite..."
echo ""

# 1. Basic Environment Tests
echo "1️⃣ Testing Environment..."
pytest tests/test_env_basic.py -v --tb=short || echo "⚠️  Some environment tests failed"
echo ""

# 2. Reward Function Tests
echo "2️⃣ Testing Reward Functions..."
pytest tests/test_reward.py -v --tb=short || echo "⚠️  Some reward tests failed"
echo ""

# 3. Database Tests
echo "3️⃣ Testing Database..."
pytest tests/test_database.py -v --tb=short || echo "⚠️  Some database tests failed"
echo ""

# 4. Production Tests
echo "4️⃣ Testing Production Module..."
pytest tests/test_production.py -v --tb=short || echo "⚠️  Some production tests failed"
echo ""

# 5. Integration Tests
echo "5️⃣ Testing Integration..."
pytest tests/test_integration.py -v --tb=short || echo "⚠️  Some integration tests failed"
echo ""

# 6. API Tests
echo "6️⃣ Testing API..."
pytest tests/test_api.py -v --tb=short || echo "⚠️  Some API tests failed"
echo ""

echo "=" 
echo "✅ Test Suite Complete!"
echo ""

# Generate coverage report (optional)
read -p "Generate coverage report? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📊 Generating coverage report..."
    pytest tests/ --cov=. --cov-report=html --cov-report=term
    echo "📄 Coverage report saved to htmlcov/index.html"
fi

echo ""
echo "🎉 Testing complete!"
