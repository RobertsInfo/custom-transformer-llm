import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("🤖 Loading Phi-3-mini (3.8 Billion Parameters)... This may take a minute...")

# 1. Download and initialize the massive pre-trained brain and tokenizer
model_id = "microsoft/Phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
# trust_remote_code is required for Phi-3's custom architecture setup
# FIXED: Added attn_implementation="eager" to bypass the RoPE configuration bug
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=False,
    attn_implementation="eager"
)


print("🧠 Model loaded successfully! Type 'quit' to exit.")

while True:
    user_input = input("\nAsk Phi-3 anything: ")
    if user_input.lower() == 'quit':
        break
    if len(user_input) == 0:
        continue

    # 2. Format the prompt into Microsoft's specific instruction layout
    messages = [{"role": "user", "content": user_input}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # 3. Tokenize the entire sentence (sub-word tokenization!)
    inputs = tokenizer(prompt, return_tensors="pt")

    print("\n--- Phi-3 Generating Response ---")

    # 4. Generate the response tokens
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7)

    # 5. Decode the numbers back into full paragraphs of text
    # We slice out the input prompt so it only prints the new answer
    generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    print(generated_text)
    print("-" * 40)
