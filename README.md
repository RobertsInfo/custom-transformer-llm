# My Custom Transformer Model & Private AI Server Workspace

A complete, production-grade Large Language Model (LLM) engineering sandbox built inside PyCharm using PyTorch, Hugging Face, ChromaDB, and FastAPI. This project documents my exact software engineering journey—starting from a raw, memoryless letter-prediction script and evolving all the way up to a secure, vector-indexed web API server endpoint running completely on my local hardware.

---

## 🚀 Project Milestones, Terminal Data, & Code Evolution

### 📁 1. Project Setup & Workspace Configuration
* **The Goal**: Set up a clean machine learning workspace inside PyCharm to isolate code files from internal packaging software libraries.
* **What I Did**: Configured a Python 3.13 virtual environment (`.venv`) to safely run packages like PyTorch and NumPy. 
* **The Fix**: Caught and corrected directory mapping mistakes, moving my core code files safely out of hidden package folders and straight into the root project directory so Git could track them properly.

### 🔤 2. The Custom Data Pipeline & Tokenization (`data.py`)
* **The Concept**: Taught the system that neural networks are just giant mathematical calculators that run equations on integers, not alphabets. 
* **What I Built**: Wrote an algorithmic scanner to sort unique characters from text, creating a character-level dictionary tracker. It maps symbols to unique numbers using a text-to-integer (`stoi`) dictionary map (e.g., translating the text phrase `"All:"` into the numeric matrix ``) and a reverse integer-to-text (`itos`) map.
* **The Error Encountered**: My original code crashed with a scary network firewall traceback: `socket.gaierror [Errno 11001]`.
* **How I Fixed It**: Stripped out Python's internet-downloading packages and injected a localized text-duplication block directly into my local script variables. This allowed me to compile my dataset file (**`input.txt`**) completely offline, bypassing the network block.

### 🧠 3. Custom Deep Learning & Brain Architecture Evolution (`model.py` & `train.py`)
* **Phase A: The Baseline Lookup Model (Memoryless)**
  * **The Problem**: This baseline script hit an unyielding performance wall at Step 600, with its training error stalling completely at a high loss metric of **1.39**. It could only output unreadable gibberish text like `hellown custom model`.
  * **Why It Failed**: It had zero memory context. It only looked at the *very last character* typed to guess the next one, completely blind to what it wrote two steps ago.
* **Phase B: Context Window Expansion (The 16-Character Lookback)**
  * **The Solution**: Upgraded the design by adding positional embeddings and a `block_size = 16` character sliding history window.
  * **The Terminal Evidence**: The moment I ran the code, my training error instantly plummeted on Step 200 from **2.99 down to 2.13**. The tracking registers locked into a tight, highly stable band between **1.74 and 1.89**.
  * **Why It Fixed It**: The model was given the ability to "look back" at a 16-character history. It could finally see entire words starting and ending, which stabilized its mathematical confidence.
* **Phase C: Stacking the Brains (Multi-Head Self-Attention)**
  * **The Solution**: Replaced the overwhelmed single attention channel with an advanced `MultiHeadAttention` class running **4 independent attention channels side-by-side**.
  * **The Terminal Evidence**: My training error absolutely shattered, plunging from the old 1.70 ceiling all the way down to an incredible **0.3604**! The output instantly transformed into beautifully punctuated English stage dialogue (`First Citizen: First, you know Caius Marcius...`).
  * **Why It Fixed It**: Splitting the attention layer into 4 parallel heads allowed different sections to track different language rules simultaneously—one head tracked spacing, one tracked character names, one tracked suffixes, and one tracked punctuation.
* **Phase D: Saving Memory to the Hard Drive (Continuous Learning)**
  * **The Solution**: Injected `torch.save` and `torch.load` logic to dump my model's learned weights directly to a hard-drive save file (**`model_weights.pt`**).
  * **The Terminal Evidence**: On a fresh training run, my starting Step 0 loss dropped from an uneducated **3.71 all the way down to a highly trained 0.1471**!
  * **Why It Fixed It**: My model no longer suffers from amnesia when PyCharm is closed. It successfully reloads its past education straight from the hard drive, acting like a video game save file.

### 💬 4. Creating the Interactive Chat Workspace (`chat.py`)
* **The Goal**: Build a standalone console window interface so I can prompt my model live without having to run a slow training sequence every time.
* **What I Built**: Programmed a persistent command-line prompt loop running in frozen evaluation mode (`model.eval()`). This saved computer memory and made text generation incredibly fast.
* **The Error Encountered**: My script crashed on launch with `TypeError: unhashable type: 'list'` inside `data.py`.
* **How I Fixed It**: Realized that because the model processes text inputs in "batches", it was sending the data as a list inside a list (`[]`). I added an explicit index slicing parameter to flatten the data structure (`outputs[inputs.input_ids.shape:]`), allowing my terminal to print responses cleanly.
* **My Real Testing Findings**: When I typed prompts like `dog` or `bread`, the model instantly matched those sub-strings and completed them with exact surrounding theatrical context. When it printed a lonely `F` at the bottom, I deduced it simply hit its hardcoded `max_new_tokens=300` character limit mid-word.

### 🌐 5. Building the Live Internet Search Agent (`search_agent.py`)
* **The Goal**: Scale up the project to use a massive pre-trained open-source brain (Microsoft's **Phi-3-mini 3.8 Billion Parameter model**) and connect it to a live internet tool to answer unscripted questions.
* **What I Built**: Integrated the `ddgs` library to execute live DuckDuckGo text searches, creating a function that grabs the top 3 internet snippets for any prompt I type.
* **The Error Encountered**: When asking about dates or current events, the model threw a hard refusal: *"I cannot provide information about future events as my training data only includes information up until September 2021."*
* **Why It Failed & How I Fixed It**: Smaller models have rigid safety instructions hardcoded into their brain weights. When it saw the year 2026, it panicked. Furthermore, Phi-3 lacks a native system-token parser, so it ignored my system prompts. I bypassed this by injecting the live search snippets directly inside the `{"role": "user"}` payload block instead.
* **My Real Testing Findings**: When I asked the search agent `who won the superbowl before sept 2021`, it actively fired its web crawler (`🌐 Searching the web...`), read the internet data, and successfully gave me the correct historical answer: **"The Tampa Bay Buccaneers won the Super Bowl... securing their victory in Super Bowl LV on February 7, 2021, against the Kansas City Chiefs."**

### 🔒 6. Ingesting Private Local Files with Offline RAG (`rag_agent.py`)
* **The Goal**: Teach the model to read private text documents directly from my computer's folder and answer questions about them completely offline, keeping my data 100% secure from the internet.
* **What I Built**: Created a `knowledge_base` folder and a private text file containing custom metrics (`company_secrets.txt`). I wrote a local file-handler script to open the text, scan for user keywords, and feed the relevant sentences straight to a local copy of Google's **Gemma-2B** model.
* **The Security Cohesion Fix**: Integrated secure environment handling (`os.getenv("HF_TOKEN")`) to cleanly authenticate my access token to Google's gated model layout without ever risking my raw password strings being leaked onto public GitHub repositories.
* **My Empirical Stress Test**: To prove the model was reading my hard drive rather than guessing or using internet knowledge, I added a completely fictional fact: *"The secret ingredient in the laboratory cafeteria soup is crushed purple Martian dust."*
* **The Terminal Evidence**: When I asked `What is the secret ingredient in the cafeteria soup?`, the script fired its local scanner (`🔍 Scanning local knowledge_base files...`), processed the math, and printed out the exact answer: **"The secret ingredient in the laboratory cafeteria soup is crushed purple Martian dust."** This gave me absolute proof that my private document loader was functioning.

### 📡 7. Deploying a Semantic Vector Database & Web API Server (`rag_server.py`)
* **The Goal**: Upgrade the local document loader to a professional, enterprise-grade architecture. Instead of just searching for basic matching words, the system needs to understand the *meaning* of sentences, and it must run as a live web server endpoint so other external applications can talk to it.
* **What I Built**:
  1. Integrated **ChromaDB** backed by `sentence-transformers` embedding calculations. This automatically converts my entire document folder into high-dimensional mathematical coordinate points.
  2. Wrapped the entire pipeline inside a **FastAPI** backend network infrastructure web server, running continuously on an open-port execution loop via **Uvicorn** (`uvicorn rag_server:app --reload`).
* **The Error Encountered**: Testing my live network API server returned a `500 Internal Server Error` in my web browser interface dashboard.
* **How I Fixed It**: Realized that Gemma-2B outputs an extra tensor structural array layer over network protocols. I adjusted my decoding line to `outputs[inputs.input_ids.shape:]`, cleanly flattening the network layers.
