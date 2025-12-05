"""
API Test Scripti
FastAPI servisini test etmek için örnek istekler
"""

import requests
import json
from typing import Dict, Any

BASE_URL = "http://localhost:8000"

def print_response(title: str, response: requests.Response):
    """Print response in formatted way"""
    print("\n" + "=" * 60)
    print(f"📌 {title}")
    print("=" * 60)
    print(f"Status Code: {response.status_code}")
    print(f"Response:")
    print(json.dumps(response.json(), indent=2, ensure_ascii=False))
    print("=" * 60)

def test_health_check():
    """Test health check endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        print_response("Health Check", response)
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: API service is not running!")
        print("   Please start the service first with 'python -m uvicorn app.main:app --reload'")
        return False
    return True

def test_distance_prediction():
    """Test distance prediction"""
    data = {
        "rssi": -75.5,
        "lqi": 105.2,
        "throughput": 20000
    }
    response = requests.post(f"{BASE_URL}/predict/distance", json=data)
    print_response("Distance Prediction", response)

def test_human_presence():
    """Test human presence detection"""
    data = {
        "rssi": -82.3,
        "lqi": 98.5,
        "throughput": 15000,
        "timestamp": 45
    }
    response = requests.post(f"{BASE_URL}/detect/human-presence", json=data)
    print_response("Human Presence Detection", response)

def test_device_location():
    """Test device location classification"""
    data = {
        "rssi": -78.2,
        "lqi": 103.5,
        "throughput": 19500,
        "rssi_stddev": 1.2
    }
    response = requests.post(f"{BASE_URL}/classify/device-location", json=data)
    print_response("Device Location Classification", response)

def test_signal_quality():
    """Test signal quality scoring"""
    data = {
        "rssi": -75.5,
        "lqi": 105.2,
        "throughput": 20000
    }
    response = requests.post(f"{BASE_URL}/score/signal-quality", json=data)
    print_response("Signal Quality Scoring", response)

def test_anomaly_detection():
    """Test anomaly detection"""
    # Normal data
    data_normal = {
        "rssi": -75.5,
        "lqi": 105.2,
        "throughput": 20000
    }
    response = requests.post(f"{BASE_URL}/detect/anomaly", json=data_normal)
    print_response("Anomaly Detection (Normal Data)", response)
    
    # Abnormal data
    data_anomaly = {
        "rssi": -95.5,
        "lqi": 50.2,
        "throughput": 500
    }
    response = requests.post(f"{BASE_URL}/detect/anomaly", json=data_anomaly)
    print_response("Anomaly Detection (Abnormal Data)", response)

def test_signal_prediction():
    """Test signal prediction"""
    data = {
        "history": [
            {"timestamp": 0, "rssi": -75, "lqi": 105, "throughput": 20000},
            {"timestamp": 15, "rssi": -78, "lqi": 103, "throughput": 19500},
            {"timestamp": 30, "rssi": -80, "lqi": 101, "throughput": 19000}
        ],
        "future_steps": 3
    }
    response = requests.post(f"{BASE_URL}/predict/signal-quality", json=data)
    print_response("Signal Quality Prediction", response)

def test_all_endpoints():
    """Test all endpoints"""
    print("\n" + "🚀" * 30)
    print("FASTAPI ML SERVICE - API TEST")
    print("🚀" * 30)
    
    # Health check
    if not test_health_check():
        return
    
    # Test all endpoints
    test_distance_prediction()
    test_human_presence()
    test_device_location()
    test_signal_quality()
    test_anomaly_detection()
    test_signal_prediction()
    
    print("\n" + "✅" * 30)
    print("ALL TESTS COMPLETED!")
    print("✅" * 30)

if __name__ == "__main__":
    test_all_endpoints()

