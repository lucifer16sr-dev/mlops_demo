
import torch
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.sentiment_classifier.model import SentimentClassifier, SimpleTokenizer
from models.utils import save_model, load_model


def test_sentiment_classifier_initialization():
    model = SentimentClassifier(
        vocab_size=1000,
        embedding_dim=64,
        hidden_dim=32,
        num_layers=1
    )
    
    assert model is not None
    assert isinstance(model, SentimentClassifier)


def test_sentiment_classifier_forward():
    model = SentimentClassifier(
        vocab_size=1000,
        embedding_dim=64,
        hidden_dim=32,
        num_layers=1
    )
    
    batch_size = 4
    seq_length = 50
    x = torch.randint(0, 1000, (batch_size, seq_length))
    
    output = model(x)
    
    assert output.shape == (batch_size, 2)  # Binary classification


def test_tokenizer():
    tokenizer = SimpleTokenizer(vocab_size=100)
    
    # Build vocab from sample texts
    texts = ["I love this", "This is great", "Not good"]
    tokenizer.build_vocab(texts)
    
    # Test encoding
    encoded = tokenizer.encode("I love this")
    assert isinstance(encoded, torch.Tensor)
    assert encoded.dim() == 1


def test_model_save_load(tmp_path):
    model = SentimentClassifier(
        vocab_size=1000,
        embedding_dim=64,
        hidden_dim=32
    )
    
    # Save model
    model_path = os.path.join(tmp_path, "test_model.pth")
    save_model(model, model_path)
    
    # Load model
    loaded_model = load_model(SentimentClassifier, model_path)
    
    assert loaded_model is not None
    assert isinstance(loaded_model, SentimentClassifier)
    
    # Test that loaded model works
    x = torch.randint(0, 1000, (2, 50))
    output = loaded_model(x)
    assert output.shape == (2, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])