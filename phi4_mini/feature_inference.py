import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from interp_algo.SAE import SAE


class Phi4FeatureInference:
    """
    Phi4 inference that returns SAE features instead of raw activations.
    SAE encoding happens directly in the hooks - no intermediate storage.

    Usage:
        phi = Phi4FeatureInference(sae_path="./weights/SAE")
        responses, tokens, features = phi.generate(["Hello!"], max_new_tokens=32)
        # features: list of (64, seq_len, 12288) tensors
    """

    def __init__(self, model_path="./weights/phi4-mini", sae_path="./weights/SAE", device="cuda"):
        self.device = device

        # load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, trust_remote_code=False
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
        self.model.eval()

        self.num_layers = len(self.model.model.layers)  # 32
        self.embed_dim = self.model.config.hidden_size  # 3072
        self.hidden_dim = self.embed_dim * 4  # 12288

        # load SAEs
        self.mlp_saes, self.att_saes = self._load_saes(sae_path)

        # feature storage (filled by hooks)
        self.features = {}

        # register hooks
        for layer_idx in range(self.num_layers):
            layer = self.model.model.layers[layer_idx]
            layer.input_layernorm.register_forward_hook(self._make_hook(layer_idx, "att"))
            layer.post_attention_layernorm.register_forward_hook(self._make_hook(layer_idx, "mlp"))

    def _load_saes(self, sae_path):
        """Load all 62 SAEs, asserting all exist."""
        mlp_saes = {}
        att_saes = {}

        # check all exist first
        missing = []
        for layer in range(self.num_layers - 1):  # mlp 0-30
            path = f"{sae_path}/layer_{layer}_mlp_sae.pt"
            if not os.path.exists(path):
                missing.append(path)
        for layer in range(1, self.num_layers):  # att 1-31
            path = f"{sae_path}/layer_{layer}_att_sae.pt"
            if not os.path.exists(path):
                missing.append(path)

        assert not missing, f"Missing SAE weights:\n" + "\n".join(missing)

        # load mlp SAEs (layers 0-30)
        for layer in range(self.num_layers - 1):
            path = f"{sae_path}/layer_{layer}_mlp_sae.pt"
            sae = SAE(self.embed_dim, self.hidden_dim)
            sae.load_state_dict(torch.load(path, weights_only=True))
            sae.to(self.device)
            sae.eval()
            mlp_saes[layer] = sae

        # load att SAEs (layers 1-31)
        for layer in range(1, self.num_layers):
            path = f"{sae_path}/layer_{layer}_att_sae.pt"
            sae = SAE(self.embed_dim, self.hidden_dim)
            sae.load_state_dict(torch.load(path, weights_only=True))
            sae.to(self.device)
            sae.eval()
            att_saes[layer] = sae

        print(f"Loaded {len(mlp_saes)} mlp SAEs + {len(att_saes)} att SAEs")
        return mlp_saes, att_saes

    def _make_hook(self, layer_idx, kind):
        """Hook that encodes activations to features immediately."""
        def hook(module, input, output):
            # output: (batch, seq, embed_dim)
            with torch.no_grad():
                if kind == "mlp" and layer_idx in self.mlp_saes:
                    feat = self.mlp_saes[layer_idx].encode(output.float())  # (batch, seq, hidden_dim)
                    self.features[f"layer_{layer_idx}_mlp"] = feat.detach()
                elif kind == "att" and layer_idx in self.att_saes:
                    feat = self.att_saes[layer_idx].encode(output.float())  # (batch, seq, hidden_dim)
                    self.features[f"layer_{layer_idx}_att"] = feat.detach()
        return hook

    def generate(self, prompts, max_new_tokens=128):
        batch_size = len(prompts)

        # format prompts
        formatted = []
        for p in prompts:
            messages = [{"role": "user", "content": p}]
            formatted.append(self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            ))

        inputs = self.tokenizer(formatted, return_tensors="pt", padding=True).to(self.device)
        generated_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        prompt_len = generated_ids.shape[1]

        # per-sequence storage
        all_tokens = [[] for _ in range(batch_size)]
        # 64 hook positions: layer*2 = att, layer*2+1 = mlp
        all_feats = [[[] for _ in range(self.num_layers * 2)] for _ in range(batch_size)]
        stopped = [False] * batch_size

        past_key_values = None

        for step in range(max_new_tokens):
            if all(stopped):
                break

            if step == 0:
                input_ids = generated_ids
            else:
                input_ids = next_token.unsqueeze(1)

            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True
                )

            past_key_values = outputs.past_key_values
            logits = outputs.logits
            next_token = logits[:, -1, :].argmax(dim=-1)

            # collect features and tokens for non-stopped sequences
            for b in range(batch_size):
                if stopped[b]:
                    continue

                tok_id = next_token[b].item()
                all_tokens[b].append(self.tokenizer.decode([tok_id]))

                for l in range(self.num_layers):
                    # att features: hook_idx = l * 2
                    att_key = f"layer_{l}_att"
                    if att_key in self.features:
                        all_feats[b][l * 2].append(self.features[att_key][b, -1, :].cpu())
                    else:
                        # no SAE for this position (att[0]), store zeros
                        all_feats[b][l * 2].append(torch.zeros(self.hidden_dim))

                    # mlp features: hook_idx = l * 2 + 1
                    mlp_key = f"layer_{l}_mlp"
                    if mlp_key in self.features:
                        all_feats[b][l * 2 + 1].append(self.features[mlp_key][b, -1, :].cpu())
                    else:
                        # no SAE for this position (mlp[31]), store zeros
                        all_feats[b][l * 2 + 1].append(torch.zeros(self.hidden_dim))

                if tok_id == self.tokenizer.eos_token_id:
                    stopped[b] = True

            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(1)], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones(batch_size, 1, device=self.device, dtype=attention_mask.dtype)], dim=1)

        # decode responses
        responses = []
        for b in range(batch_size):
            responses.append(self.tokenizer.decode(generated_ids[b, prompt_len:], skip_special_tokens=True))

        # stack features
        features = []
        for b in range(batch_size):
            seq_len = len(all_tokens[b])
            if seq_len == 0:
                features.append(torch.zeros(self.num_layers * 2, 0, self.hidden_dim))
                continue

            stacked = [torch.stack(all_feats[b][i]) for i in range(self.num_layers * 2)]
            features.append(torch.stack(stacked))  # (64, seq_len, 12288)

        # responses: [str, ...] length B
        # all_tokens: [[str, ...], ...] length B
        # features: [Tensor(64, seq_len, 12288), ...] length B
        return responses, all_tokens, features


if __name__ == "__main__":
    phi = Phi4FeatureInference()
    prompts = ["What is 2+2?", "Name a fruit."]
    responses, tokens, features = phi.generate(prompts, max_new_tokens=32)

    print(f"Responses: {responses}")
    print(f"Tokens: {tokens}")
    print(f"Features: {[f.shape for f in features]}")  # [(64, seq_len, 12288), ...]

    # check sparsity
    for i, feat in enumerate(features):
        active = (feat > 0.5).float().mean().item()
        print(f"  Sequence {i}: {active*100:.2f}% features active (>0.5)")
