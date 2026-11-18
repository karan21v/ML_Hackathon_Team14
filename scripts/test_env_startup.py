import os
import sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, BASE_DIR)

from src.api.main import health, list_trips, get_trip

def main():
    print("Testing server with env vars:")
    print("PATHSNAP_MODEL_PATH:", os.environ.get("PATHSNAP_MODEL_PATH"))
    print("PATHSNAP_REQUIRE_ML:", os.environ.get("PATHSNAP_REQUIRE_ML"))
    print("PATHSNAP_TRIPS_ZIP:", os.environ.get("PATHSNAP_TRIPS_ZIP"))
    print("PATHSNAP_TRIP_PARTITION:", os.environ.get("PATHSNAP_TRIP_PARTITION"))

    h = health()
    print("\n[Health Check]:", h)
    assert h["status"] == "ONLINE"

    trips = list_trips()
    print(f"\n[Available Trips]: {len(trips)} trips found")
    for t in trips[:6]:
        print(f"  * {t['trip_id']}: {t['description']} ({t['num_points']} points)")

    first_trip = trips[0]["trip_id"]
    t_data = get_trip(first_trip)
    print(f"\n[Trip Detail Check]: Loaded {len(t_data['points'])} points from {first_trip}")
    print("SUCCESS: Environment variables, model, and ZIP dataset verified!")

if __name__ == "__main__":
    main()
