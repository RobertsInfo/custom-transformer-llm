import os
import torch

file_path = "input.txt"

# A clean, offline backup text dataset
backup_text = """
First Citizen: Before we proceed any further, hear me speak.
All: Speak, speak.
First Citizen: You are all resolved rather to die than to famish?
All: Resolved, resolved.
First Citizen: First, you know Caius Marcius is chief enemy to the people.
All: We know't, we know't.
First Citizen: Let us revenge this with our pikes, ere we become rakes: 
for the gods know I speak this in hunger for bread, not in thirst for revenge.
Second Citizen: Would you proceed especially against Caius Marcius?
All: Against him first: he's a very dog to the commonalty.
"""

# Multiply the text so the model has enough data to learn patterns
text = backup_text * 500

# Save it locally so the script can read it standardly
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(text)

# Read the text file
with open(file_path, 'r', encoding='utf-8') as f:
    text = f.read()

# Setup vocab and token maps
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Convert to PyTorch tensor
data = torch.tensor(encode(text), dtype=torch.long)
