import os
import torch
from fastapi import FastAPI
from pydantic import BaseModel
import chromadb
from chromadb.utils import embedding_functions
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- 1. INITIALIZE WEB SERVER & VECTOR DATABASE ---
app = FastAPI(title="Secure Local AI Server")
HF_TOKEN = os.getenv("HF_TOKEN", "hf_YOUR_TOKEN_HERE")
KNOWLEDGE_FOLDER = "knowledge_base"

# Setup local vector storage directory
chroma_client = chromadb.PersistentClient(path="./chroma_db")
# Using a local, lightweight embedding model to convert text sentences into coordinate matrices
default_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
collection = chroma_client.get_or_create_collection(name="private_vault", embedding_function=default_ef)


# --- 2. VECTOR STORAGE: INGEST AND SEMANTIC INDEX ---
def index_knowledge_base():
    """Reads local text files, breaks them down, and builds a vector semantic index."""
    if not os.path.exists(KNOWLEDGE_FOLDER):
        print("📁 Knowledge folder not found. Skipping vector initialization.")
        return

    print("🔍 Vector Database: Ingesting and indexing documents...")
    documents = []
    ids = []

    doc_id = 0
    for filename in os.listdir(KNOWLEDGE_FOLDER):
        if filename.endswith(".txt"):
            file_path = os.path.join(KNOWLEDGE_FOLDER, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f.readlines():
                    line = line.strip()
                    if line and len(line) > 10:  # Skip empty or tiny lines
                        documents.append(line)
                        ids.append(f"doc_{doc_id}")
                        doc_id += 1

    if documents:
        # Upsert indexes the documents into mathematical coordinate space
        collection.upsert(documents=documents, ids=ids)
        print(f"✅ Successfully indexed {len(documents)} text blocks into Vector DB.")


# Build the vector library indexes right at startup
index_knowledge_base()

# --- 3. LOAD OPEN-SOURCE PARAMS ---
model_id = "google/gemma-2b-it"
print("🤖 Loading neural matrices into system memory...")
tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    model_id, token=HF_TOKEN, trust_remote_code=False, attn_implementation="eager"
)
print("🧠 Local LLM Brain loaded safely!")


# --- 4. API CONTRACT INITIALIZATION ---
class ChatRequest(BaseModel):
    question: str


@app.post("/v1/chat")
def secure_chat_endpoint(request: ChatRequest):
    """Network endpoint processing semantic retrieval and output matrix generation."""
    user_query = request.question

    # Semantic Search: Query database for context based on MEANING, not just words
    results = collection.query(query_texts=[user_query], n_results=1)

    # Fallback structure if database returns blank matrices
    private_context = results['documents'][0][0] if results['documents'] and results['documents'][
        0] else "No metrics located."

    # Package into explicit chat format payload
    user_payload = (
        f"You are a secure corporate assistant web server. Use only these private internal notes to answer:\n"
        f"--- NOTES ---\n{private_context}\n---------------\n\n"
        f"Question: {user_query}"
    )

    messages = [{"role": "user", "content": user_payload}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=150)

    # FIXED: Added explicit index mapping parameters to flatten network tensor layers
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    return {"query": user_query, "response": response.strip(), "source_context": private_context}
