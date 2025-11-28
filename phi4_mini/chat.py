from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_path = "./weights/phi4-mini"
model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=False)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
model.eval()

print("Model loaded. Type 'quit' to exit.\n")

conversation = []

while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break
    
    # append to history
    conversation.append({"role": "user", "content": user_input})
    
    # format full conversation
    prompt = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True  # KV cache for faster generation
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    # append assistant response to history
    conversation.append({"role": "assistant", "content": response})
    
    print(f"Phi4: {response}\n")