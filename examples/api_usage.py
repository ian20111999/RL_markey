#!/usr/bin/env python3
"""
Example: Using the Production API

This script demonstrates how to interact with the REST API:
- Making predictions
- Listing models
- Getting model information
- Managing models
"""

import requests
import json
import numpy as np

# API Configuration
API_BASE_URL = "http://localhost:8000"


def check_health():
    """Check API health"""
    print("🔍 Checking API Health...")
    response = requests.get(f"{API_BASE_URL}/health")
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ API is healthy")
        print(f"   Models Loaded: {data['models_loaded']}")
        print(f"   Registry Size: {data['registry_size']}")
        print(f"   Production Models: {data['production_models']}")
    else:
        print(f"❌ API health check failed: {response.status_code}")
    print()


def list_models():
    """List all available models"""
    print("📋 Listing Models...")
    response = requests.get(f"{API_BASE_URL}/models")
    
    if response.status_code == 200:
        models = response.json()
        print(f"Found {len(models)} models:\n")
        
        for i, model in enumerate(models[:5], 1):  # Show top 5
            print(f"{i}. {model['model_id']}")
            print(f"   Algorithm: {model['algorithm']}")
            print(f"   Score: {model['profitability_score']:.1f}/100")
            print(f"   Mean PnL: {model['mean_pnl']:+.2f}")
            print(f"   Production Ready: {'✅' if model['production_ready'] else '❌'}")
            print()
    else:
        print(f"❌ Failed to list models: {response.status_code}")
    print()


def get_best_model():
    """Get the best performing model"""
    print("🏆 Getting Best Model...")
    response = requests.get(f"{API_BASE_URL}/models/best/current")
    
    if response.status_code == 200:
        model = response.json()
        print(f"Best Model: {model['model_id']}")
        print(f"  Algorithm: {model['algorithm']}")
        print(f"  Score: {model['profitability_score']:.1f}/100")
        print(f"  Mean PnL: {model['mean_pnl']:+.2f}")
        print(f"  Win Rate: {model['win_rate']*100:.1f}%")
        print(f"  Sharpe: {model['sharpe_ratio']:.2f}")
    else:
        print(f"❌ Failed to get best model: {response.status_code}")
    print()


def make_prediction():
    """Make a prediction using the best model"""
    print("🔮 Making Prediction...")
    
    # Create a sample observation (adjust size based on your environment)
    # This is a dummy observation - replace with real market data
    observation = np.random.randn(17).tolist()  # Assuming 17 features
    
    payload = {
        "observation": observation,
        "deterministic": True
    }
    
    response = requests.post(
        f"{API_BASE_URL}/predict",
        json=payload
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Prediction successful!")
        print(f"   Model Used: {result['model_id']}")
        print(f"   Version: {result['version']}")
        print(f"   Action: {result['action']}")
    else:
        print(f"❌ Prediction failed: {response.status_code}")
        print(f"   Error: {response.text}")
    print()


def get_leaderboard():
    """Get model performance leaderboard"""
    print("📊 Getting Leaderboard...")
    response = requests.get(f"{API_BASE_URL}/metrics/leaderboard?limit=5")
    
    if response.status_code == 200:
        models = response.json()
        print(f"Top {len(models)} Models:\n")
        print(f"{'Rank':<5} {'Model ID':<25} {'Score':>7} {'PnL':>10} {'Win%':>7}")
        print("-" * 60)
        
        for i, model in enumerate(models, 1):
            model_id = model['model_id'][:23]
            score = model['profitability_score']
            pnl = model['mean_pnl']
            win_rate = model['win_rate'] * 100
            
            print(f"{i:<5} {model_id:<25} {score:>7.1f} {pnl:>10.2f} {win_rate:>7.1f}")
    else:
        print(f"❌ Failed to get leaderboard: {response.status_code}")
    print()


def load_specific_model(model_id: str):
    """Preload a specific model into memory"""
    print(f"📥 Loading model: {model_id}...")
    response = requests.post(f"{API_BASE_URL}/models/{model_id}/load")
    
    if response.status_code == 200:
        print(f"✅ Model loaded successfully")
    else:
        print(f"❌ Failed to load model: {response.status_code}")
    print()


def main():
    print("=" * 80)
    print("🌐 PRODUCTION API USAGE EXAMPLE")
    print("=" * 80)
    print()
    print(f"API Base URL: {API_BASE_URL}")
    print()
    print("Make sure the API server is running:")
    print("  python production/cli.py serve --port 8000")
    print()
    print("=" * 80)
    print()
    
    try:
        # 1. Check health
        check_health()
        
        # 2. List models
        list_models()
        
        # 3. Get best model
        get_best_model()
        
        # 4. Get leaderboard
        get_leaderboard()
        
        # 5. Make prediction
        make_prediction()
        
        print("=" * 80)
        print("✅ API USAGE EXAMPLE COMPLETE")
        print()
        print("For more API endpoints, visit:")
        print(f"  {API_BASE_URL}/docs")
        print()
        print("=" * 80)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to API server")
        print("   Please start the API server first:")
        print("   python production/cli.py serve --port 8000")
        print()


if __name__ == "__main__":
    main()
