import os
import torch
import torch.nn.functional as F
from data import vocab_size, encode, decode
from model import MultiHeadLM

# 1. Configuration (Must exactly match train.py)
block_size = 16
embed_size = 64
num_heads = 4
weights_file = 'model_weights.pt'

# 2. Initialize and load the model instantly
model = MultiHeadLM(vocab_size, embed_size, block_size, num_heads)

if os.path.exists(weights_file):
    model.load_state_dict(torch.load(weights_file, weights_only=True))
    model.eval()  # Put model in evaluation mode
    print("🧠 Model loaded successfully! Type 'quit' to exit.")
else:
    print("❌ Error: model_weights.pt not found. Run train.py first.")
    exit()

# 3. Interactive Prompt Loop
while True:
    user_prompt = input("\nEnter a starting prompt (e.g., 'All:', 'First Citizen:'): ")
    if user_prompt.lower() == 'quit':
        break

    if len(user_prompt) == 0:
        continue

    # Convert your typing into numbers
    encoded_prompt = encode(user_prompt)
    context = torch.tensor([encoded_prompt], dtype=torch.long)

    print("\n--- Model Generating Response ---")
    # Generate 300 characters based on what you typed
    generated_tokens = model.generate(context, max_new_tokens=300).tolist()[0]

    # Print the result cleanly
    print(decode(generated_tokens))
    print("-" * 40)
