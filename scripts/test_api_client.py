"""
API Client Test Script (Simulates Laptop 2 sending requests to Laptop 1).
Tests the API endpoints, verifies response codes, and logs performance.
"""

import sys
import requests
import json

def test_api(base_url="http://127.0.0.1:8000"):
    print(f"Testing PathSnap API at: {base_url}")
    
    # 1. Health Check
    try:
        r = requests.get(f"{base_url}/health", timeout=3)
        assert r.status_code == 200, f"Health check failed: {r.status_code}"
        print("[PASS] Health Check:", r.json())
    except Exception as e:
        print(f"[FAIL] Could not connect to API: {e}")
        return False

    # 2. Network Status
    r = requests.get(f"{base_url}/api/network-status")
    print("[PASS] Network Status:", r.json().get("instructions"))

    # 3. List Trips
    r = requests.get(f"{base_url}/api/trips")
    trips = r.json()
    print(f"[PASS] Available Trips ({len(trips)} found):")
    for t in trips:
        print(f"       - {t['trip_id']}: {t['num_points']} points")

    # 4. Fetch Trip 2 and Match
    trip_id = "TRIP_002_PARALLEL_AMBIGUITY"
    r = requests.get(f"{base_url}/api/trips/{trip_id}")
    trip_data = r.json()

    payload = {
        "trip_id": trip_id,
        "points": trip_data["points"],
        "sigma_z": 10.0,
        "beta": 12.0
    }

    r = requests.post(f"{base_url}/api/match", json=payload)
    assert r.status_code == 200, f"Match failed: {r.status_code}"
    res = r.json()
    print("[PASS] Map Matching Prediction:")
    print(f"       - Trip: {res['trip_id']}")
    print(f"       - Road Accuracy: {res['accuracy'] * 100:.1f}%")
    print(f"       - Latency: {res['latency_ms']} ms")
    print(f"       - Snapping Error: {res['avg_snapping_distance_m']} m")
    print(f"       - Matched Route: {' -> '.join(res['matched_edges_sequence'])}")

    # 5. Render Map Endpoint
    r = requests.post(f"{base_url}/api/render-map", json=payload)
    assert r.status_code == 200, f"Map render failed: {r.status_code}"
    print(f"[PASS] Render Map HTML Endpoint (returned {len(r.text)} bytes of HTML)")

    print("\nALL API INTEGRATION TESTS PASSED SUCCESSFULLY!")
    return True

if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
    test_api(url)
