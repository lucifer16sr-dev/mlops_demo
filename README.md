# MLOps Inference Platform Demo

A production-ready MLOps inference platform demonstrating model serving, monitoring, and deployment automation.

## 🎯 Project Overview

This project showcases:
- **Multi-model serving** with Ray Serve
- **REST API** with FastAPI
- **Containerization** with Docker
- **Monitoring** with Prometheus & Grafana
- **CI/CD** pipelines
- **Evaluation** frameworks

## 📁 Project Structure

mlops_inference_platform/
├── models/ # ML models and training scripts
├── serving/ # Model serving layer (Ray Serve, FastAPI)
├── monitoring/ # Metrics and observability
├── evaluation/ # Testing and evaluation
├── tests/ # Test suite
└── README.md


## 🚀 Quick Start

### 1. Setup Environment

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

## 📦 Models

### Sentiment Classifier
A PyTorch-based LSTM model for binary sentiment classification (positive/negative).

**Architecture:**
- Embedding layer (vocab_size × embedding_dim)
- Bidirectional LSTM (2 layers)
- Fully connected layers with dropout
- Binary classification output

**Training:**
python models/sentiment_classifier/train.py

**Model Location:**
- Trained model: `models/sentiment_classifier/checkpoint.pth`
- Model class: `models/sentiment_classifier/model.py`
- Training script: `models/sentiment_classifier/train.py`

**Usage:**
from models.sentiment_classifier.model import SentimentClassifier
from models.utils import load_model

**Testing Inference:**
python test_inference.py

model = load_model(SentimentClassifier, "models/sentiment_classifier/checkpoint.pth")
# Use model for inference...