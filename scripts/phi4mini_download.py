from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "microsoft/Phi-4-mini-instruct"
save_path = "./weights/phi4-mini"

# download and save
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=False)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)

model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

# later, load from local
model = AutoModelForCausalLM.from_pretrained(save_path, dtype=torch.bfloat16, device_map="auto", trust_remote_code=False)
tokenizer = AutoTokenizer.from_pretrained(save_path, trust_remote_code=False)