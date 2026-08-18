# Custom Multi-Head Transformer LLM From Scratch

A miniature, character-level Large Language Model (LLM) built using PyTorch inside PyCharm. This repository documents the exact architectural evolution, optimization stages, and implementation of a generative transformer pipeline.

## 🚀 Project Milestones & Discoveries

### 📁 Project Architecture & Workspace
* **Isolated Environment**: Configured a Python 3.13 virtual environment (`.venv`) inside PyCharm to isolate dependencies like PyTorch and NumPy.
* **Root Directory Discipline**: Organized core files safely outside internal environment directories to maintain clean workspace structure.

### 🔤 Data Pipeline & Tokenization (`data.py`)
* **Vocabulary Extraction**: Created an algorithmic scanner to extract and sort unique characters from text data.
* **Bi-directional Mapping**: Built character-to-integer (`stoi`) and integer-to-character (`itos`) dictionary lookups for matrix translation.
* **Air-Gapped Training Data**: Deployed an offline text engine to bypass local network limits, localizing standard text sequences into `input.txt`.

### 🧠 Neural Network Evolution (`model.py`)
* **Baseline Lookup Model**: Built a standard embedding layer that hit a performance ceiling at a high loss of **1.39** due to an absence of context memory.
* **Contextual Window Expansion**: Upgraded the design with position embeddings and a `block_size = 16` lookback window, dropping and stabilizing loss to a tight **1.74 to 1.89** band.
* **Multi-Head Self-Attention**: Engineered a parallel head layout running 4 individual attention heads side-by-side to track multiple linguistic patterns at once.

### ⚙️ Optimization & Continuous Learning (`train.py`)
* **Backpropagation Loop**: Implemented an AdamW optimization cycle running across 5,000 steps to adjust matrix weights based on error gradients.
* **Weight Persistence Engine**: Integrated persistent file handling to save model matrices directly to a local binary file (**`model_weights.pt`**).
* **State Check Verification**: Confirmed active loading after verifying a training launch loss drop from **3.71 down to 0.1471**, validating persistent memory retention.

### 💬 The Production Interface (`chat.py`)
* **Evaluation Mode Execution**: Frozen backpropagation updates using `model.eval()` to prioritize execution speed and reduce RAM usage.
* **Persistent Chat Loop**: Established an interactive runtime panel in the PyCharm terminal to feed custom prompts (`dog`, `bread`, `Second Citizen:`) into the model for text generation.
