
import requests
import time

BASE_URL = "http://localhost:8080"

def test_metrics():
    print("Testing metrics endpoint...")
    
    # Make some requests to generate metrics
    for i in range(5):
        try:
            response = requests.post(
                f"{BASE_URL}/predict/sentiment_classifier",
                json={"text": f"Test message {i}"}
            )
            print(f"Request {i+1}: Status {response.status_code}")
        except Exception as e:
            print(f"Request {i+1} failed: {e}")
        time.sleep(0.5)
    
    # Check metrics endpoint
    try:
        response = requests.get(f"{BASE_URL}/metrics")
        print(f"\nMetrics endpoint status: {response.status_code}")
        print("\nSample metrics output:")
        print(response.text[:500])
    except Exception as e:
        print(f"Failed to fetch metrics: {e}")

if __name__ == "__main__":
    test_metrics()