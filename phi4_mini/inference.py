import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class Phi4Inference:
    def __init__(self, model_path="./weights/phi4-mini", device="cuda"):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, trust_remote_code=False
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
        self.model.eval()
        self.num_layers = len(self.model.model.layers)  # 32
        self.embed_dim = self.model.config.hidden_size  # 3072
        self.activations = {}  # {f"layer_{l}_att_in": (batch, seq, embed), ...}
        self.device = device
        
        for layer_idx in range(self.num_layers):
            layer = self.model.model.layers[layer_idx]
            layer.input_layernorm.register_forward_hook(self._make_hook(layer_idx, "att_in"))
            layer.post_attention_layernorm.register_forward_hook(self._make_hook(layer_idx, "mlp_in"))
    
    def _make_hook(self, layer_idx, name):
        def hook(module, input, output):
            # output: (batch, seq, embed_dim)
            self.activations[f"layer_{layer_idx}_{name}"] = output.detach()
        return hook
    
    def generate(self, prompts, max_new_tokens=128):
        batch_size = len(prompts)  # B
        
        # format prompts
        formatted = []  # list of B strings
        for p in prompts:
            messages = [{"role": "user", "content": p}]
            formatted.append(self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ))
        
        inputs = self.tokenizer(formatted, return_tensors="pt", padding=True).to(self.device)
        generated_ids = inputs["input_ids"]  # (B, prompt_len)
        attention_mask = inputs["attention_mask"]  # (B, prompt_len)
        prompt_len = generated_ids.shape[1]
        
        # per-sequence storage
        all_tokens = [[] for _ in range(batch_size)]  # B x [list of token strings]
        all_acts = [[[] for _ in range(self.num_layers * 2)] for _ in range(batch_size)]  # B x 64 x [list of (embed,) tensors]
        stopped = [False] * batch_size  # B bools
        
        past_key_values = None  # KV cache, starts empty
        
        for step in range(max_new_tokens):
            if all(stopped):
                break
            
            # first step: process full prompt, later: just the new token
            if step == 0:
                input_ids = generated_ids  # (B, prompt_len)
            else:
                input_ids = next_token.unsqueeze(1)  # (B, 1)
            
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True
                )
            
            past_key_values = outputs.past_key_values  # updated KV cache
            logits = outputs.logits  # (B, seq_len, vocab) where seq_len=prompt_len or 1
            next_token = logits[:, -1, :].argmax(dim=-1)  # (B,)
            
            # collect activations and tokens for non-stopped sequences
            for b in range(batch_size):
                if stopped[b]:
                    continue
                
                tok_id = next_token[b].item()
                all_tokens[b].append(self.tokenizer.decode([tok_id]))
                
                for l in range(self.num_layers):
                    # activations are (B, seq_len, embed), grab last position for this batch
                    all_acts[b][l * 2].append(self.activations[f"layer_{l}_att_in"][b, -1, :].cpu())  # (embed,)
                    all_acts[b][l * 2 + 1].append(self.activations[f"layer_{l}_mlp_in"][b, -1, :].cpu())  # (embed,)
                
                if tok_id == self.tokenizer.eos_token_id:
                    stopped[b] = True
            
            # extend ids and mask for next iteration
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(1)], dim=1)  # (B, prompt_len+step+1)
            attention_mask = torch.cat([attention_mask, torch.ones(batch_size, 1, device=self.device, dtype=attention_mask.dtype)], dim=1)  # (B, prompt_len+step+1)
        
        # decode full responses
        responses = []  # B strings
        for b in range(batch_size):
            responses.append(self.tokenizer.decode(generated_ids[b, prompt_len:], skip_special_tokens=True))
        
        # stack activations: list of B tensors, each (num_layers*2, seq_len, embed_dim)
        activations = []
        for b in range(batch_size):
            seq_len = len(all_tokens[b])
            if seq_len == 0:
                activations.append(torch.empty(self.num_layers * 2, 0, self.embed_dim))
                continue
            # all_acts[b][i] is list of seq_len tensors of shape (embed,)
            stacked = [torch.stack(all_acts[b][i]) for i in range(self.num_layers * 2)]  # 64 x (seq_len, embed)
            activations.append(torch.stack(stacked))  # (64, seq_len, embed)
        
        return responses, all_tokens, activations


if __name__ == "__main__":
    phi = Phi4Inference()
    prompts = ["What is 2+2?", "Name a fruit."]
    responses, tokens, acts = phi.generate(prompts, max_new_tokens=32)
    
    print(f"Responses: {responses}")
    print(f"Tokens: {tokens}")
    print(f"Activations: {[a.shape for a in acts]}")  # [(64, seq_len_0, 3072), (64, seq_len_1, 3072)]