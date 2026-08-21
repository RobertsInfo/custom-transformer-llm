import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ddgs import DDGS

# 1. Initialize the Search Engine
ddgs = DDGS()


def internet_search(query: str) -> str:
    """Fetches real-time snippets from the live internet using an updated context engine."""
    print(f"🌐 Searching the web for: '{query}'...")
    try:
        # Utilizing the modern context manager syntax for duckduckgo_search
        with DDGS() as s:
            results = [r for r in s.text(query, max_results=3)]

        if not results:
            return "No relevant web results found."

        context_block = ""
        for i, res in enumerate(results):
            context_block += f"[{i + 1}] Title: {res['title']}\nSnippet: {res['body']}\n\n"
        return context_block
    except Exception as e:
        return f"Search error encountered: {e}"

# 2. Initialize Phi-3 Model & Tokenizer
model_id = "microsoft/Phi-3-mini-4k-instruct"
print("🤖 Loading model layers...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=False,
    attn_implementation="eager"
)
print("🧠 System initialized!")

# 3. Interactive Search Agent Loop
while True:
    user_query = input("\nAsk the Search Agent anything: ")
    if user_query.lower() == 'quit':
        break
    if len(user_query.strip()) == 0:
        continue

    # Step A: Retrieve web context
    web_context = internet_search(user_query)

    # Step B: Inject context directly into the User role (Phi-3 standard)
    user_payload = (
        f"Context from web search:\n"
        f"{web_context}\n"
        f"Question: {user_query}\n\n"
        f"Instructions: Answer the question using the web context provided above. "
        f"Be concise and factual."
    )

    messages = [
        {"role": "user", "content": user_payload}
    ]

    # Step C: Format using native chat template
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt")

    print("🧠 Synthesizing live web data...")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200)

    # Step D: Decode and print only the generated tokens
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print("\n--- Agent Response ---")
    print(response.strip())
    print("-" * 40)
