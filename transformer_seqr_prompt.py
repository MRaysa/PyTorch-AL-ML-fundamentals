import torch
import torch.nn as nn
import torch.optim as optim

# ============================================================================
# DEVICE SETUP
# ============================================================================

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}\n")

# ============================================================================
# SIMPLE TRANSFORMER - REVERSES SEQUENCES
# ============================================================================

class SequenceReverser(nn.Module):
    def __init__(self, vocab_size=20, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, 50, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=128,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        # x shape: (batch, seq_len)
        seq_len = x.size(1)
        x = self.embedding(x) + self.pos_embedding[:, :seq_len, :]
        x = self.transformer(x)
        x = self.fc_out(x)
        return x


# ============================================================================
# TRAINING
# ============================================================================

def train_model():
    vocab_size = 20
    seq_length = 8
    
    model = SequenceReverser(vocab_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    print("Task: Learn to reverse number sequences")
    print(f"Example: [3, 7, 2, 9] → [9, 2, 7, 3]\n")
    print("Training...\n")
    
    # Training loop
    for epoch in range(2000):
        model.train()
        total_loss = 0
        
        # Generate 20 random sequences per epoch
        for _ in range(20):
            # Random sequence: numbers 1-19 (0 is padding)
            input_seq = torch.randint(1, vocab_size, (1, seq_length)).to(device)
            
            # Target: reversed sequence
            target_seq = torch.flip(input_seq, dims=[1])
            
            # Forward pass
            output = model(input_seq)
            loss = criterion(output.view(-1, vocab_size), target_seq.view(-1))
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Print progress
        if (epoch + 1) % 400 == 0:
            avg_loss = total_loss / 20
            print(f"Epoch {epoch+1}/2000 | Loss: {avg_loss:.4f}")
    
    print("\n✓ Training complete!\n")
    return model


# ============================================================================
# TESTING
# ============================================================================

def test_model(model):
    model.eval()
    
    print("=" * 60)
    print("TESTING THE MODEL")
    print("=" * 60)
    
    with torch.no_grad():
        for i in range(8):
            # Create test sequence
            test_input = torch.randint(1, 20, (1, 8)).to(device)
            
            # Get prediction
            output = model(test_input)
            predicted = output.argmax(dim=-1)
            
            # Expected output (reversed)
            expected = torch.flip(test_input, dims=[1])
            
            # Display
            input_nums = test_input[0].cpu().tolist()
            pred_nums = predicted[0].cpu().tolist()
            expected_nums = expected[0].cpu().tolist()
            
            match = "✓ CORRECT" if pred_nums == expected_nums else "✗ WRONG"
            
            print(f"\nTest {i+1}:")
            print(f"  Input:    {input_nums}")
            print(f"  Expected: {expected_nums}")
            print(f"  Predicted: {pred_nums}")
            print(f"  {match}")


# ============================================================================
# INTERACTIVE MODE
# ============================================================================

def interactive_mode(model):
    print("\n" + "=" * 60)
    print("INTERACTIVE MODE")
    print("=" * 60)
    print("Enter a sequence of numbers (1-19) separated by spaces")
    print("Example: 3 7 2 9 5 1 8 4")
    print("Type 'quit' to exit\n")
    
    model.eval()
    
    while True:
        user_input = input("Your sequence: ").strip()
        
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        
        if not user_input:
            continue
        
        try:
            # Parse input
            numbers = [int(x) for x in user_input.split()]
            
            # Validate numbers
            if any(n < 1 or n > 19 for n in numbers):
                print("Error: Numbers must be between 1 and 19\n")
                continue
            
            if len(numbers) > 15:
                print("Error: Maximum 15 numbers allowed\n")
                continue
            
            # Convert to tensor
            input_tensor = torch.tensor([numbers]).to(device)
            
            # Get prediction
            with torch.no_grad():
                output = model(input_tensor)
                predicted = output.argmax(dim=-1)
            
            # Expected (reversed)
            expected = numbers[::-1]
            predicted_nums = predicted[0].cpu().tolist()
            
            # Display results
            print(f"\n  Input:     {numbers}")
            print(f"  Expected:  {expected}")
            print(f"  Predicted: {predicted_nums}")
            
            if predicted_nums == expected:
                print("  ✓ CORRECT!\n")
            else:
                print("  ✗ Not quite right\n")
                
        except ValueError:
            print("Error: Please enter valid numbers separated by spaces\n")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SIMPLE TRANSFORMER - SEQUENCE REVERSAL")
    print("=" * 60)
    print()
    
    # Train
    trained_model = train_model()
    
    # Test
    test_model(trained_model)
    
    # Interactive
    interactive_mode(trained_model)
    
    print("\n" + "=" * 60)