
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
import re


class SentimentClassifier(nn.Module):
    
    def __init__(
        self,
        vocab_size: int = 10000,
        embedding_dim: int = 128,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
        num_classes: int = 2
    ):
        super(SentimentClassifier, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim * 2, 64)  # *2 for bidirectional
        self.fc2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, num_classes)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor):
        # Embedding
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use the last hidden state (concatenate forward and backward)
        # hidden shape: (num_layers * 2, batch_size, hidden_dim)
        last_hidden = hidden[-2:, :, :].permute(1, 0, 2)  # (batch_size, 2, hidden_dim)
        last_hidden = last_hidden.contiguous().view(last_hidden.size(0), -1)  # (batch_size, hidden_dim * 2)
        
        # Fully connected layers
        out = self.dropout(F.relu(self.fc1(last_hidden)))
        out = self.dropout(F.relu(self.fc2(out)))
        out = self.output(out)
        
        return out
    
    def predict(self, text: str, tokenizer, device: str = 'cpu'):
        self.eval()
        with torch.no_grad():
            # Tokenize and encode
            encoded = tokenizer(text)
            if isinstance(encoded, list):
                encoded = torch.tensor([encoded], dtype=torch.long)
            else:
                encoded = encoded.unsqueeze(0) if encoded.dim() == 1 else encoded
            
            encoded = encoded.to(device)
            
            # Forward pass
            logits = self.forward(encoded)
            probs = F.softmax(logits, dim=1)
            
            # Get probabilities
            prob_negative = probs[0][0].item()
            prob_positive = probs[0][1].item()
            
            prediction = "positive" if prob_positive > prob_negative else "negative"
            confidence = max(prob_positive, prob_negative)
            
            return {
                "prediction": prediction,
                "confidence": confidence,
                "probabilities": {
                    "negative": prob_negative,
                    "positive": prob_positive
                }
            }


class SimpleTokenizer:
    
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocab_built = False
    
    def build_vocab(self, texts: List[str]):
        word_counts = {}
        
        # Count words
        for text in texts:
            words = self._tokenize(text)
            for word in words:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Sort by frequency and take top vocab_size
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Build mappings (reserve 0 for padding, 1 for unknown)
        self.word_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_word = {0: '<PAD>', 1: '<UNK>'}
        
        for idx, (word, count) in enumerate(sorted_words[:self.vocab_size - 2], start=2):
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word
        
        self.vocab_built = True
        print(f"Vocabulary built with {len(self.word_to_idx)} words")
    
    def _tokenize(self, text: str):
        # Simple tokenization: lowercase, split on whitespace
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text.split()
    
    def encode(self, text: str, max_length: int = 128):
        if not self.vocab_built:
            raise ValueError("Vocabulary not built. Call build_vocab() first.")
        
        words = self._tokenize(text)
        indices = [self.word_to_idx.get(word, 1) for word in words]  # 1 is <UNK>
        
        # Pad or truncate
        if len(indices) > max_length:
            indices = indices[:max_length]
        else:
            indices = indices + [0] * (max_length - len(indices))
        
        return torch.tensor(indices, dtype=torch.long)
    
    def __call__(self, text: str):
        return self.encode(text)