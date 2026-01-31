
import requests
import json
import time


def test_health_check(base_url: str = "http://localhost:8000"):
    print("Testing health check...")
    response = requests.get(f"{base_url}/health_check")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def test_single_prediction(base_url: str = "http://localhost:8000"):
    print("Testing single prediction...")
    
    test_text = "I love this product! It's amazing!"
    
    response = requests.post(
        f"{base_url}/predict",
        json={"text": test_text}
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def test_batch_prediction(base_url: str = "http://localhost:8000"):
    print("Testing batch prediction...")
    
    test_texts = [
        "I love this product!",
        "This is terrible.",
        "Great service!",
        "Poor quality."
    ]
    
    requests_list = [{"text": text} for text in test_texts]
    
    response = requests.post(
        f"{base_url}/predict_batch",
        json={"requests": requests_list}
    )
    
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()


def test_performance(base_url: str = "http://localhost:8000", num_requests: int = 10):
    print(f"Testing performance with {num_requests} requests...")
    
    test_text = "This is a test sentence for performance testing."
    start_time = time.time()
    
    for i in range(num_requests):
        response = requests.post(
            f"{base_url}/predict",
            json={"text": f"{test_text} Request {i+1}"}
        )
        if response.status_code != 200:
            print(f"Request {i+1} failed: {response.status_code}")
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    print(f"Total time: {elapsed:.2f}s")
    print(f"Average time per request: {elapsed/num_requests:.3f}s")
    print(f"Throughput: {num_requests/elapsed:.2f} requests/second")
    print()


if __name__ == "__main__":
    base_url = "http://localhost:8000"
    
    print("=" * 60)
    print("Testing Ray Serve Deployment")
    print("=" * 60)
    print()
    
    try:
        test_health_check(base_url)
        test_single_prediction(base_url)
        test_batch_prediction(base_url)
        test_performance(base_url, num_requests=10)
        
        print("=" * 60)
        print("All tests completed!")
        print("=" * 60)
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to Ray Serve.")
        print("Make sure Ray Serve is running:")
        print("  python serving/start_ray_serve.py")
    except Exception as e:
        print(f"Error: {e}")