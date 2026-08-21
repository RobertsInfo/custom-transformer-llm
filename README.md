## Custom Multi-Head Transformer & Open-Source LLM Workspace
A complete, production-ready Large Language Model (LLM) engineering sandbox built using PyTorch and Hugging Face inside PyCharm. This repository documents the exact architectural evolution, optimization stages, and implementation from a baseline memoryless transformer to a 3.8-billion parameter open-source assistant.
## 🚀 Project Milestones & Discoveries## 📁 1. Project Architecture & Workspace

* Isolated Environment: Configured a Python 3.13 virtual environment (.venv) inside PyCharm to isolate dependencies like PyTorch and NumPy.
* Root Directory Discipline: Organized core files safely outside internal environment directories to maintain clean workspace structure.

## 🔤 2. Data Pipeline & Tokenization (data.py)

* Vocabulary Extraction: Created an algorithmic scanner to extract and sort unique characters from text data.
* Bi-directional Mapping: Built character-to-integer (stoi) and integer-to-character (itos) dictionary lookups for matrix translation.
* Air-Gapped Training Data: Deployed an offline text engine to bypass local network limits (socket.gaierror [Errno 11001]), localizing standard text sequences into input.txt.

## 🧠 3. Neural Network Evolution (model.py & train.py)

* Baseline Lookup Model: Built a standard embedding layer that hit a performance ceiling at a high loss of 1.39 due to an absence of context memory.
* Contextual Window Expansion: Upgraded the design with position embeddings and a block_size = 16 lookback window, dropping and stabilizing loss to a tight 1.74 to 1.89 band.
* Multi-Head Self-Attention: Engineered a parallel head layout running 4 individual attention heads side-by-side to track multiple linguistic patterns at once, driving loss down to 0.3604.
* Backpropagation Loop: Implemented an AdamW optimization cycle running across 5,000 steps to adjust matrix weights based on error gradients.
* Weight Persistence Engine: Integrated persistent file handling to save model matrices directly to a local binary file (model_weights.pt).
* State Check Verification: Confirmed active loading after verifying a training launch loss drop from 3.71 down to 0.1471, validating persistent memory retention.

## 💬 4. The Production Interface (chat.py)

* Evaluation Mode Execution: Froze backpropagation updates using model.eval() to prioritize execution speed and reduce RAM usage.
* Causal Mask Inference: Built a persistent runtime panel in the PyCharm terminal to feed custom prompts (dog, bread, Second Citizen:) into the model for text generation.

## 🤖 5. Production Scaling with Microsoft Phi-3 (phi_chat.py)

* Hugging Face Framework Link: Integrated the transformers library to load pre-trained, multi-billion parameter foundation models directly into the workspace.
* Bypassing Remote Configuration Conflicts: Resolved a KeyError: 'type' crash by migrating from remote script execution (trust_remote_code=True) to the native codebase (trust_remote_code=False) with eager attention.
* Unscripted Global Reasoning: Deployed a local chat pipeline capable of answering unscripted real-world inquiries (geography, coding, science) from its 3.8-billion pre-trained weights without local training files.

## 🌐 6. Live Web Search Agent Architecture (search_agent.py)

* Internet Tool Integration: Integrated the duckduckgo-search library wrapper inside your workspace to serve as an on-demand external search tool.
* Network Query Function: Programmed an isolated internet_search function with built-in network safety blocks (try/except) to scrape the top 3 live text snippets for any prompt you type.
* Contextual Data Merging: Constructed a multi-layered user string block (user_payload) that catches raw internet snippets and injects them directly alongside your questions.

## 🕵️‍♂️ 7. Algorithmic Debugging & Breakthroughs

* Overriding Safety Safeguards: Discovered that smaller 3.8B foundation models contain rigid, hardcoded refusal scripts regarding their knowledge cutoff limits.
* Role Alignment Refactoring: Discovered that Phi-3 ignores classic {"role": "system"} tags because it lacks a native system-token parser. Bypassed this limit completely by injecting your live web data directly into the active {"role": "user"} array, forcing the model to read the data.
* Context Synthesis Victory: Tested and verified that your script successfully bypassed temporal amnesia to cleanly read, analyze, and print out live, unscripted historical milestones (such as the Tampa Bay Buccaneers game statistics) entirely from the scraped web context.

## 📁 8. Version Control & Portfolio Discipline

* Strategic Code Freezing: Applied a standard engineering rule ("If it works, don't touch it") to preserve your active runtime script architecture, opting to leave runtime warnings alone since the data pipeline is functional.
* Git Synchronization: Successfully staged, committed, and pushed your complete search_agent.py script to your private GitHub repository, backing up your work safely online.




