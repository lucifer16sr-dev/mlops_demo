
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from models.sentiment_classifier.model import SentimentClassifier, SimpleTokenizer
from models.utils import save_model


class SentimentDataset(Dataset):    
    def __init__(self, texts: List[str], labels: List[int], tokenizer: SimpleTokenizer, max_length: int = 128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoded = self.tokenizer.encode(text, self.max_length)
        return encoded, torch.tensor(label, dtype=torch.long)


def generate_sample_data(num_samples: int = 1000):
    positive_samples = [
        "I love this product, it's amazing!",
        "Great service, highly recommend.",
        "Excellent quality, very satisfied.",
        "This is fantastic, works perfectly.",
        "Wonderful experience, thank you!",
        "Outstanding performance, exceeded expectations.",
        "Really happy with this purchase.",
        "Top notch quality and service.",
        "Impressive features and design.",
        "Best product I've ever used."
    ]
    
    negative_samples = [
        "Terrible product, very disappointed.",
        "Poor quality, waste of money.",
        "Not worth it, broken after one use.",
        "Awful experience, would not recommend.",
        "Very bad service and support.",
        "Disappointing quality, expected better.",
        "Not satisfied with this purchase.",
        "Low quality materials used.",
        "Doesn't work as advertised.",
        "Regret buying this product."
    ]
    
    texts = []
    labels = []
    
    for _ in range(num_samples):
        if torch.rand(1).item() > 0.5:
            texts.append(positive_samples[torch.randint(0, len(positive_samples), (1,)).item()])
            labels.append(1)
        else:
            texts.append(negative_samples[torch.randint(0, len(negative_samples), (1,)).item()])
            labels.append(0)
    
    return texts, labels


def train_model(
    model: SentimentClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 10,
    device: str = 'cpu',
    learning_rate: float = 0.001
):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for texts, labels in train_loader:
            texts = texts.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for texts, labels in val_loader:
                texts = texts.to(device)
                labels = labels.to(device)
                
                outputs = model(texts)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"  ✓ New best model (val_loss: {best_val_loss:.4f})")
    
    return model


def main():
    print("=" * 60)
    print("Training Sentiment Classifier")
    print("=" * 60)
    
    # Configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    batch_size = 32
    num_epochs = 10
    vocab_size = 10000
    embedding_dim = 128
    hidden_dim = 64
    num_layers = 2
    
    # Generate sample data
    print("\n1. Generating sample data...")
    texts, labels = generate_sample_data(num_samples=2000)
    
    # Build tokenizer
    print("\n2. Building tokenizer...")
    tokenizer = SimpleTokenizer(vocab_size=vocab_size)
    tokenizer.build_vocab(texts)
    
    # Create datasets
    print("\n3. Creating datasets...")
    split_idx = int(0.8 * len(texts))
    train_texts, train_labels = texts[:split_idx], labels[:split_idx]
    val_texts, val_labels = texts[split_idx:], labels[split_idx:]
    
    train_dataset = SentimentDataset(train_texts, train_labels, tokenizer)
    val_dataset = SentimentDataset(val_texts, val_labels, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    
    # Initialize model
    print("\n4. Initializing model...")
    model = SentimentClassifier(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers
    )
    
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    print("\n5. Training model...")
    trained_model = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=num_epochs,
        device=device
    )
    
    # Save model
    print("\n6. Saving model...")
    model_dir = os.path.join(os.path.dirname(__file__))
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "checkpoint.pth")
    
    metadata = {
        'vocab_size': vocab_size,
        'embedding_dim': embedding_dim,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'tokenizer': tokenizer  # Save tokenizer for inference
    }
    
    save_model(trained_model, model_path, metadata)
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Model saved to: {model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()