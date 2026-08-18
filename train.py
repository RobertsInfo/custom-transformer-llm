import os
import torch
from data import data, vocab_size
from model import MultiHeadLM

# Hyperparameters (Keep these exactly the same)
batch_size = 4
block_size = 16
embed_size = 64
num_heads = 4

# Initialize model
model = MultiHeadLM(vocab_size, embed_size, block_size, num_heads)

# --- CONTINUOUS TRAINING LOGIC ---
weights_file = 'model_weights.pt'
if os.path.exists(weights_file):
    print("Found existing weights! Loading progress...")
    # Load the saved numbers back into the model layers
    model.load_state_dict(torch.load(weights_file, weights_only=True))
else:
    print("No past weights found. Starting training from scratch...")

# Initialize optimizer AFTER loading weights
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# ... (keep your existing get_batch helper function here) ...

def get_batch():
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

# Training loop
print("Starting training loop...")
for steps in range(5000):  # You can run this for 1,000 steps at a time
    xb, yb = get_batch()
    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if steps % 200 == 0:
        print(f"Step {steps}: Loss {loss.item():.4f}")

# Save the updated progress at the end of the loop
torch.save(model.state_dict(), weights_file)
print("Updated weights saved to disk!")

# --- CUSTOM PROMPT GENERATION ---
from data import encode, decode

# 1. Type your custom starting text here
starting_text = "First Citizen:"

# 2. Convert your text prompt into token numbers
encoded_prompt = encode(starting_text)
context = torch.tensor([encoded_prompt], dtype=torch.long)

# 3. Tell the model to generate 500 characters based on your prompt
print(f"\n--- Model Completing Prompt: '{starting_text}' ---")
generated_tokens = model.generate(context, max_new_tokens=500).tolist()

# 4. Print the final results
print(decode(generated_tokens[0]))

