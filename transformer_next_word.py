import torch
import torch.nn as nn
import torch.optim as optim

# ============================================================================
# DEVICE SETUP
# ============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

# ============================================================================
# SIMPLE TRANSFORMER - NEXT WORD PREDICTION
# ============================================================================

class NextWordPredictor(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, 100, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            batch_first=True,
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        seq_len = x.size(1)
        x = self.embedding(x) + self.pos_embedding[:, :seq_len, :]
        x = self.transformer(x)
        x = self.fc_out(x)
        return x


# ============================================================================
# TRAINING DATA
# ============================================================================

SENTENCES = [
    "the cat sits on the mat",
    "the dog runs in the park",
    "i love to eat rice",
    "she reads a good book",
    "he plays in the garden",
    "the sun shines very bright",
    "birds fly in the sky",
    "fish swim in the water",
    "children play with toys",
    "we go to the school",
    "the moon comes at night",
    "trees grow in the forest",
    "flowers bloom in spring",
    "rain falls from the clouds",
    "the cat drinks some milk",
    "dogs like to run fast",
    "people live in the city",
    "cars move on the road",
    "the train goes very fast",
    "boats float on the river",
]

def build_vocab(sentences):
    words = set()
    for sent in sentences:
        words.update(sent.split())
    
    vocab = {'<PAD>': 0}
    for word in sorted(words):
        vocab[word] = len(vocab)
    
    return vocab

def prepare_data(sentences, vocab):
    data = []
    for sent in sentences:
        words = sent.split()
        indices = [vocab[w] for w in words]
        
        # Create input-target pairs
        for i in range(len(indices) - 1):
            input_seq = indices[:i+1]
            target = indices[i+1]
            data.append((input_seq, target))
    
    return data


# ============================================================================
# TRAINING
# ============================================================================

def train_model(vocab):
    vocab_size = len(vocab)
    train_data = prepare_data(SENTENCES, vocab)
    
    model = NextWordPredictor(vocab_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Task: Predict the next word in a sentence")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Training examples: {len(train_data)}\n")
    print("Training...\n")
    
    # Training loop
    for epoch in range(1000):
        model.train()
        total_loss = 0
        
        for input_seq, target in train_data:
            # Pad input sequence
            padded = input_seq + [0] * (10 - len(input_seq))
            padded = padded[:10]
            
            input_tensor = torch.tensor([padded]).to(device)
            target_tensor = torch.tensor([target]).to(device)
            
            # Forward pass
            output = model(input_tensor)
            # Get prediction at last position of actual sequence
            loss = criterion(output[:, len(input_seq)-1:len(input_seq), :].squeeze(1), target_tensor)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Print progress
        if (epoch + 1) % 200 == 0:
            avg_loss = total_loss / len(train_data)
            print(f"Epoch {epoch+1}/1000 | Loss: {avg_loss:.4f}")
    
    print("\n✓ Training complete!\n")
    return model


# ============================================================================
# PREDICTION
# ============================================================================

def predict_next_word(model, text, vocab, top_k=3):
    model.eval()
    
    # Convert text to indices
    words = text.lower().strip().split()
    if not words:
        return []
    
    indices = []
    for w in words:
        if w in vocab:
            indices.append(vocab[w])
        else:
            print(f"Warning: '{w}' not in vocabulary, skipping...")
    
    if not indices:
        return []
    
    # Pad sequence
    padded = indices + [0] * (10 - len(indices))
    padded = padded[:10]
    
    input_tensor = torch.tensor([padded]).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        # Get predictions at last word position
        predictions = output[0, len(indices)-1, :]
        
        # Get top-k predictions
        probs = torch.softmax(predictions, dim=0)
        top_probs, top_indices = torch.topk(probs, min(top_k, len(vocab)))
        
        # Convert back to words
        idx_to_word = {v: k for k, v in vocab.items()}
        results = []
        for prob, idx in zip(top_probs, top_indices):
            word = idx_to_word.get(idx.item(), '<UNK>')
            if word != '<PAD>':
                results.append((word, prob.item()))
        
        return results


# ============================================================================
# TESTING
# ============================================================================

def test_model(model, vocab):
    print("=" * 60)
    print("AUTOMATIC TESTING")
    print("=" * 60)
    
    test_cases = [
        "the cat sits on the",
        "i love to eat",
        "the dog runs in the",
        "fish swim in the",
        "the sun shines very",
    ]
    
    for test in test_cases:
        predictions = predict_next_word(model, test, vocab, top_k=3)
        print(f"\nInput: '{test}'")
        print(f"Top predictions:")
        for i, (word, prob) in enumerate(predictions, 1):
            print(f"  {i}. {word} ({prob*100:.1f}%)")


# ============================================================================
# INTERACTIVE MODE
# ============================================================================

def interactive_mode(model, vocab):
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE")
    print("=" * 60)
    print("Enter a sentence and I'll predict the next word!")
    print("Type 'quit' to exit\n")
    
    while True:
        text = input("Your input: ").strip()
        
        if text.lower() == 'quit':
            print("Goodbye!")
            break
        
        if not text:
            continue
        
        predictions = predict_next_word(model, text, vocab, top_k=5)
        
        if predictions:
            print("\nNext word predictions:")
            for i, (word, prob) in enumerate(predictions, 1):
                print(f"  {i}. {word} ({prob*100:.1f}%)")
            print()
        else:
            print("Sorry, couldn't make a prediction. Try words from the training set.\n")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("NEXT WORD PREDICTION TRANSFORMER")
    print("=" * 60)
    print()
    
    # Build vocabulary
    vocab = build_vocab(SENTENCES)
    
    print("Training sentences:")
    for i, sent in enumerate(SENTENCES[:5], 1):
        print(f"  {i}. {sent}")
    print(f"  ... and {len(SENTENCES)-5} more\n")
    
    # Train
    trained_model = train_model(vocab)
    
    # Test
    test_model(trained_model, vocab)
    
    # Interactive
    interactive_mode(trained_model, vocab)