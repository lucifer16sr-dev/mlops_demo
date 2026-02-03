
import pytest
import requests
import json
from typing import Dict, Any


BASE_URL = "http://localhost:8080"


def test_root_endpoint():
    response = requests.get(f"{BASE_URL}/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "version" in data


def test_health_check():
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model" in data


def test_list_models():
    response = requests.get(f"{BASE_URL}/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert "total" in data


def test_single_prediction():
    payload = {"text": "I love this product!"}
    response = requests.post(
        f"{BASE_URL}/predict/sentiment_classifier",
        json=payload
    )
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "confidence" in data
    assert data["prediction"] in ["positive", "negative"]


def test_batch_prediction():
    payload = {
        "texts": [
            "I love this product!",
            "This is terrible."
        ]
    }
    response = requests.post(
        f"{BASE_URL}/predict/sentiment_classifier/batch",
        json=payload
    )
    assert response.status_code == 200
    data = response.json()
    assert "predictions" in data
    assert "total" in data
    assert len(data["predictions"]) == 2


def test_invalid_model():
    payload = {"text": "Test"}
    response = requests.post(
        f"{BASE_URL}/predict/invalid_model",
        json=payload
    )
    assert response.status_code == 404


def test_swagger_docs():
    response = requests.get(f"{BASE_URL}/docs")
    assert response.status_code == 200


def test_openapi_schema():
    response = requests.get(f"{BASE_URL}/openapi.json")
    assert response.status_code == 200
    schema = response.json()
    assert "openapi" in schema
    assert "info" in schema


if __name__ == "__main__":
    pytest.main([__file__, "-v"])