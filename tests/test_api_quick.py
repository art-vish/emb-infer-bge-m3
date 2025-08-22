#!/usr/bin/env python3
"""
Quick API test for emb-infer-bge-m3
Tests the simplified single endpoint with selective vector generation
"""

import requests
import json
import time
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
BASE_URL = "http://localhost:8000"
API_TOKEN = os.getenv("API_TOKEN", "test_token_bge_m3_2024")

HEADERS = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_TOKEN}"
}

def test_health():
    """Test health endpoint"""
    print("🏥 Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health", headers=HEADERS)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Status: {data.get('status')}")
            print(f"   📊 Model: {data.get('model')}")
            return True
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ❌ Health check error: {e}")
        return False

def test_embedding_endpoint_basic():
    """Test basic embedding endpoint"""
    payload = {
        "input": "Test text for embedding generation"
    }
    
    response = requests.post(f"{BASE_URL}/v1/embeddings", headers=HEADERS, json=payload)
    assert response.status_code == 200
    
    data = response.json()
    assert "data" in data
    assert len(data["data"]) > 0

def _test_embedding_endpoint(test_name, payload):
    """Test embedding endpoint with specific payload"""
    print(f"\n🧪 {test_name}")
    try:
        start_time = time.time()
        response = requests.post(f"{BASE_URL}/v1/embeddings", headers=HEADERS, json=payload)
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Success! Time: {response_time:.2f}s")
            print(f"   📊 Vector types: {data.get('embedding_types', [])}")
            print(f"   📝 Items processed: {len(data.get('data', []))}")
            return True
        else:
            print(f"   ❌ Failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def main():
    print("🚀 QUICK API TEST - emb-infer-bge-m3")
    print("=" * 50)
    
    # Test health first
    if not test_health():
        print("\n❌ Health check failed. Make sure the service is running.")
        print("Run: docker-compose up -d")
        return False
    
    print("\n📝 TESTING EMBEDDING ENDPOINT")
    print("-" * 30)
    
    test_cases = [
        {
            "name": "All vectors (default)",
            "payload": {
                "input": "Test text for embedding generation"
            }
        },
        {
            "name": "Dense only",
            "payload": {
                "input": "Semantic search query",
                "return_dense": True,
                "return_sparse": False,
                "return_colbert": False
            }
        },
        {
            "name": "Sparse only", 
            "payload": {
                "input": "Keyword matching text",
                "return_dense": False,
                "return_sparse": True,
                "return_colbert": False
            }
        },
        {
            "name": "Batch processing",
            "payload": {
                "input": [
                    "First document to process",
                    "Second document to process",
                    "Third document to process"
                ],
                "return_dense": True,
                "return_sparse": True,
                "return_colbert": False
            }
        }
    ]
    
    results = []
    for test_case in test_cases:
        result = _test_embedding_endpoint(test_case["name"], test_case["payload"])
        results.append(result)
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results) + (1 if test_health() else 0)  # +1 for health check
    total = len(results) + 1
    
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("✨ API is working correctly with simplified architecture")
        print("🔗 Available endpoints:")
        print("   - POST /v1/embeddings (BGE-M3 selective vectors)")
        print("   - GET /health (Health check)")
        print("   - GET /stats (Performance statistics)")
        print("   - GET /v1/models (Model information)")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        print("Check the service logs: docker-compose logs -f")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
