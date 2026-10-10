import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- HUGGING FACE SECURITY COHESION ---
# Fallback hardcoded token from your token-fix step if the environment variable isn't set yet
HF_TOKEN = os.getenv("HF_TOKEN", "hf_YOUR_TOKEN_HERE")

KNOWLEDGE_FOLDER = "knowledge_base"


def retrieve_relevant_chunks(user_query: str) -> str:
    """Scans local text files to pull paragraphs matching the user's keywords."""
    print("🔍 Scanning local knowledge_base files...")
    matched_chunks = []

    if not os.path.exists(KNOWLEDGE_FOLDER):
        return "Knowledge base folder not found."

    for filename in os.listdir(KNOWLEDGE_FOLDER):
        if filename.endswith(".txt"):
            file_path = os.path.join(KNOWLEDGE_FOLDER, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    query_words = user_query.lower().split()
                    if any(word in line.lower() for word in query_words if len(word) > 3):
                        matched_chunks.append(line)

    if not matched_chunks:
        return "No relevant internal data snippets discovered."

    return "\n".join(matched_chunks[:3])


# --- SYSTEM INITIALIZATION WITH AUTHENTICATION ---
model_id = "google/gemma-2b-it"  # Note: ensure you use the '-it' conversational model for chat template safety
print("🤖 Loading pre-trained model brains from local disk...")

# We inject token=HF_TOKEN directly into the download handlers
tokenizer = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    token=HF_TOKEN,
    trust_remote_code=False,
    attn_implementation="eager"
)
print("🧠 Private RAG system initialized cleanly!")

# Interactive RAG Chat Loop
while True:
    user_query = input("\nAsk your private document scanner (or type 'exit' to quit): ")
    if user_query.lower() in ['quit', 'exit']:
        print("👋 Powering down private RAG system. Goodbye!")
        break
    if len(user_query.strip()) == 0:
        continue

    # 1. Retrieve data facts offline
    private_context = retrieve_relevant_chunks(user_query)

    # 2. Package context into Gemma's native User block structure
    user_payload = (
        f"You are a secure assistant. Use only these private internal notes to answer the question:\n"
        f"--- NOTES ---\n{private_context}\n---------------\n\n"
        f"Question: {user_query}"
    )

    messages = [{"role": "user", "content": user_payload}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")

    print("🧠 Processing private matrices...")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=150)

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print("\n--- Private Document Response ---")
    print(response.strip())
    print("-" * 40)
