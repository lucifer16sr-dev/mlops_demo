# test_inference.py (create in project root)
import torch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from models.sentiment_classifier.model import SentimentClassifier, SimpleTokenizer
from models.utils import load_model

def test_inference():
    print("=" * 60)
    print("Testing Sentiment Classifier Inference")
    print("=" * 60)
    
    # Load checkpoint
    model_path = "models/sentiment_classifier/checkpoint.pth"
    
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found at {model_path}")
        print("Please train the model first: python models/sentiment_classifier/train.py")
        return
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Reconstruct model using saved parameters
    model_params = checkpoint.get('model_params', checkpoint.get('metadata', {}))
    model_init_params = {k: v for k, v in model_params.items() 
                        if k not in ['tokenizer'] and v is not None}
    model = SentimentClassifier(**model_init_params)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load tokenizer from metadata
    metadata = checkpoint.get('metadata', {})
    model_params_dict = checkpoint.get('model_params', {})
    tokenizer = metadata.get('tokenizer') or model_params_dict.get('tokenizer')
    
    if not tokenizer:
        print("Warning: Tokenizer not found in checkpoint metadata.")
        print("This may cause issues with inference.")
        return
    
    # Test cases
    test_texts = [
        "I love this product! It's amazing!",
        "This is terrible. Very disappointed.",
        "Great service, highly recommend.",
        "Poor quality, waste of money.",
        "Outstanding performance, exceeded expectations."
    ]
    
    print("\nRunning inference tests:\n")
    for text in test_texts:
        try:
            result = model.predict(text, tokenizer, device='cpu')
            print(f"Text: {text}")
            print(f"  → Prediction: {result['prediction']}")
            print(f"  → Confidence: {result['confidence']:.2%}")
            print(f"  → Probabilities: negative={result['probabilities']['negative']:.3f}, positive={result['probabilities']['positive']:.3f}")
            print()
        except Exception as e:
            print(f"Error with text '{text}': {e}")
            print()
    
    print("=" * 60)
    print("Inference test completed!")
    print("=" * 60)

if __name__ == "__main__":
    test_inference()