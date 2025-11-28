import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class Phi4Inference:
    def __init__(self, model_path="./weights/phi4-mini", layers=[8, 16, 24, 31], device="cuda"):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, trust_remote_code=False
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False, fix_mistral_regex = True)
        self.model.eval()
        self.layers = layers
        self.activations = {}
        self.device = device
        # register hooks
        for layer_idx in layers:
            mlp = self.model.model.layers[layer_idx].mlp
            mlp.register_forward_hook(self._make_hook(layer_idx))
    
    def _make_hook(self, layer_idx):
        def hook(module, input, output):
            # input[0]: pre-MLP, output: post-MLP
            # only keep the NEW tokens (generated), not the prompt
            self.activations[f"layer_{layer_idx}_pre"] = input[0].detach()
            self.activations[f"layer_{layer_idx}_post"] = output.detach()
        return hook
    
    def generate(self, prompts, max_new_tokens=128):
        """
        Args:
            prompts: list of strings (user questions)
            max_new_tokens: how many tokens to generate
        
        Returns:
            responses: list of strings
            activations: (batch, seq_len, num_layers*2, embed_dim)
        """
        # format as chat
        formatted = []
        for p in prompts:
            messages = [{"role": "user", "content": p}]
            formatted.append(self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ))
        
        # tokenize
        inputs = self.tokenizer(formatted, return_tensors="pt", padding=True).to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]
        
        # starts empty for each layer
        all_pre = {l: [] for l in self.layers}
        all_post = {l: [] for l in self.layers}
        
        # generate token by token to collect activations
        generated_ids = inputs["input_ids"].clone().to(self.device)
        attention_mask = inputs["attention_mask"].clone().to(self.device)

        # before the loop
        stopped = torch.zeros(generated_ids.shape[0], dtype=torch.bool, device=self.device)

        # inside the loop
        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.model(
                    input_ids=generated_ids,
                    attention_mask=attention_mask,
                )
            
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=-1)
            
            for l in self.layers:
                pre = self.activations[f"layer_{l}_pre"][:, -1:, :].cpu()
                post = self.activations[f"layer_{l}_post"][:, -1:, :].cpu()
                # if already stopped set added activation to 0 vector
                pre[stopped] = 0
                post[stopped] = 0
                all_pre[l].append(pre)
                all_post[l].append(post)
            
            stopped = stopped | (next_token.squeeze(-1) == self.tokenizer.eos_token_id)
            if stopped.all():
                break

        responses = []
        for i in range(generated_ids.shape[0]):
            response = self.tokenizer.decode(generated_ids[i, prompt_len:], skip_special_tokens=True)
            responses.append(response)
        
        # stack activations: (batch, seq_len, num_layers*2, embed_dim)
        stacked = []
        for l in self.layers:
            pre = torch.cat(all_pre[l], dim=1)   # (batch, seq_len, embed_dim)
            post = torch.cat(all_post[l], dim=1) # (batch, seq_len, embed_dim)
            stacked.append(pre)
            stacked.append(post)
        
        # (batch, seq_len, num_layers*2, embed_dim)
        activations = torch.stack(stacked, dim=2)
        
        return responses, activations


# test
if __name__ == "__main__":
    phi = Phi4Inference(layers=[8, 16, 24, 31])
    
    prompts = ["What is 2+2?", "Name a fruit."]
    responses, acts = phi.generate(prompts, max_new_tokens=32)
    
    print(f"Responses: {responses}")
    print(f"Activations shape: {acts.shape}")
    # expected: (2, ~32, 8, 3072) = (batch, seq, layers*2, embed_dim)
    print(f"EOS token id: {phi.tokenizer.eos_token_id}")