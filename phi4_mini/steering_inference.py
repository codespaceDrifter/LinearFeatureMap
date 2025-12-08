import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from interp_algo.SAE import SAE


class Phi4SteeringInference:
    """
    Phi4 inference with feature steering capability.

    Can inject/modify SAE features at any position, decode back to activations,
    and continue generation with the modified representations.

    Usage:
        phi = Phi4SteeringInference()

        # normal generation
        responses, tokens, features = phi.generate(["Hello!"], max_new_tokens=32)

        # steered generation - modify features at specific positions
        phi.set_override(layer=15, kind="mlp", token_idx=3, feature_idx=1234, value=5.0)
        phi.set_override(layer=15, kind="mlp", token_idx=3, feature_idx=5678, value=0.0)
        responses, tokens, features = phi.generate_steered(["Hello!"], max_new_tokens=32)

        phi.clear_overrides()
    """

    def __init__(self, model_path="./weights/phi4-mini", sae_path="./weights/SAE", device="cuda"):
        self.device = device

        # load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, trust_remote_code=False
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

        # steering overrides: {(layer, kind, token_idx): tensor of shape (hidden_dim,)}
        self.overrides = {}

        # current token position (tracked during generation)
        self.current_token_idx = 0
        self.steering_mode = False

        # register hooks
        for layer_idx in range(self.num_layers):
            layer = self.model.model.layers[layer_idx]
            layer.input_layernorm.register_forward_hook(self._make_hook(layer_idx, "att"))
            layer.post_attention_layernorm.register_forward_hook(self._make_hook(layer_idx, "mlp"))

    def _load_saes(self, sae_path):
        """Load all 62 SAEs in fp16."""
        mlp_saes = {}
        att_saes = {}

        missing = []
        for layer in range(self.num_layers - 1):
            path = f"{sae_path}/layer_{layer}_mlp_sae.pt"
            if not os.path.exists(path):
                missing.append(path)
        for layer in range(1, self.num_layers):
            path = f"{sae_path}/layer_{layer}_att_sae.pt"
            if not os.path.exists(path):
                missing.append(path)

        assert not missing, f"Missing SAE weights:\n" + "\n".join(missing)

        for layer in range(self.num_layers - 1):
            path = f"{sae_path}/layer_{layer}_mlp_sae.pt"
            sae = SAE(self.embed_dim, self.hidden_dim)
            sae.load_state_dict(torch.load(path, weights_only=True))
            sae.half().to(self.device)
            sae.eval()
            mlp_saes[layer] = sae

        for layer in range(1, self.num_layers):
            path = f"{sae_path}/layer_{layer}_att_sae.pt"
            sae = SAE(self.embed_dim, self.hidden_dim)
            sae.load_state_dict(torch.load(path, weights_only=True))
            sae.half().to(self.device)
            sae.eval()
            att_saes[layer] = sae

        print(f"Loaded {len(mlp_saes)} mlp SAEs + {len(att_saes)} att SAEs")
        return mlp_saes, att_saes

    def _make_hook(self, layer_idx, kind):
        """Hook that encodes to features, and optionally decodes overrides back."""
        def hook(module, input, output):
            # output: (batch, seq, embed_dim)
            with torch.no_grad():
                sae = None
                if kind == "mlp" and layer_idx in self.mlp_saes:
                    sae = self.mlp_saes[layer_idx]
                elif kind == "att" and layer_idx in self.att_saes:
                    sae = self.att_saes[layer_idx]

                if sae is None:
                    return None  # no SAE for this position

                # encode to features
                feat = sae.encode(output.half())  # (batch, seq, hidden_dim)
                self.features[f"layer_{layer_idx}_{kind}"] = feat.detach()

                # check for steering override
                if self.steering_mode:
                    override_key = (layer_idx, kind, self.current_token_idx)
                    if override_key in self.overrides:
                        # get the override feature vector
                        override_feat = self.overrides[override_key].half().to(self.device)  # (hidden_dim,)

                        # decode back to activation space
                        # we only override the last token position (current generation)
                        modified_output = output.clone()
                        decoded = sae.decode(override_feat.unsqueeze(0))  # (1, embed_dim)
                        modified_output[:, -1, :] = decoded.to(output.dtype)

                        return modified_output

                return None  # don't modify output
        return hook

    def set_override(self, layer, kind, token_idx, feature_idx=None, value=None, feature_vector=None):
        """
        Set a steering override.

        Either set a single feature:
            set_override(layer=15, kind="mlp", token_idx=3, feature_idx=1234, value=5.0)

        Or set the entire feature vector:
            set_override(layer=15, kind="mlp", token_idx=3, feature_vector=my_tensor)
        """
        key = (layer, kind, token_idx)

        if feature_vector is not None:
            self.overrides[key] = feature_vector.clone()
        else:
            # initialize from zeros or existing
            if key not in self.overrides:
                self.overrides[key] = torch.zeros(self.hidden_dim)
            self.overrides[key][feature_idx] = value

    def modify_override(self, layer, kind, token_idx, feature_idx, value):
        """Modify a single feature in an existing override."""
        key = (layer, kind, token_idx)
        if key not in self.overrides:
            self.overrides[key] = torch.zeros(self.hidden_dim)
        self.overrides[key][feature_idx] = value

    def clear_overrides(self):
        """Clear all steering overrides."""
        self.overrides = {}

    def get_override(self, layer, kind, token_idx):
        """Get the override vector for a position, or None if not set."""
        key = (layer, kind, token_idx)
        return self.overrides.get(key, None)

    def generate(self, prompts, max_new_tokens=128):
        """Normal generation without steering."""
        self.steering_mode = False
        return self._generate_internal(prompts, max_new_tokens)

    def generate_steered(self, prompts, max_new_tokens=128):
        """Generation with steering overrides applied."""
        self.steering_mode = True
        result = self._generate_internal(prompts, max_new_tokens)
        self.steering_mode = False
        return result

    def _generate_internal(self, prompts, max_new_tokens):
        batch_size = len(prompts)

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

        all_tokens = [[] for _ in range(batch_size)]
        all_feats = [[[] for _ in range(self.num_layers * 2)] for _ in range(batch_size)]
        stopped = [False] * batch_size

        past_key_values = None
        self.current_token_idx = 0

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

            for b in range(batch_size):
                if stopped[b]:
                    continue

                tok_id = next_token[b].item()
                all_tokens[b].append(self.tokenizer.decode([tok_id]))

                for l in range(self.num_layers):
                    att_key = f"layer_{l}_att"
                    if att_key in self.features:
                        all_feats[b][l * 2].append(self.features[att_key][b, -1, :].cpu())
                    else:
                        all_feats[b][l * 2].append(torch.zeros(self.hidden_dim))

                    mlp_key = f"layer_{l}_mlp"
                    if mlp_key in self.features:
                        all_feats[b][l * 2 + 1].append(self.features[mlp_key][b, -1, :].cpu())
                    else:
                        all_feats[b][l * 2 + 1].append(torch.zeros(self.hidden_dim))

                if tok_id == self.tokenizer.eos_token_id:
                    stopped[b] = True

            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(1)], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones(batch_size, 1, device=self.device, dtype=attention_mask.dtype)], dim=1)
            self.current_token_idx += 1

        responses = []
        for b in range(batch_size):
            responses.append(self.tokenizer.decode(generated_ids[b, prompt_len:], skip_special_tokens=True))

        features = []
        for b in range(batch_size):
            seq_len = len(all_tokens[b])
            if seq_len == 0:
                features.append(torch.zeros(self.num_layers * 2, 0, self.hidden_dim))
                continue

            stacked = [torch.stack(all_feats[b][i]) for i in range(self.num_layers * 2)]
            features.append(torch.stack(stacked))

        return responses, all_tokens, features


if __name__ == "__main__":
    phi = Phi4SteeringInference()

    # normal generation
    print("=== Normal generation ===")
    responses, tokens, features = phi.generate(["What is 2+2?"], max_new_tokens=16)
    print(f"Response: {responses[0]}")
    print(f"Tokens: {tokens[0]}")

    # find top features at token 0, layer 15 mlp
    feat = features[0][31, 0, :]  # layer 15 mlp = hook 31
    top_feats = feat.topk(5)
    print(f"\nTop 5 features at token 0, L15 mlp:")
    for idx, val in zip(top_feats.indices.tolist(), top_feats.values.tolist()):
        print(f"  f{idx}: {val:.2f}")

    # steering example: boost a random feature
    print("\n=== Steered generation (boosting top feature to 10.0) ===")
    top_feat_idx = top_feats.indices[0].item()
    phi.set_override(layer=15, kind="mlp", token_idx=0, feature_idx=top_feat_idx, value=10.0)
    responses2, tokens2, features2 = phi.generate_steered(["What is 2+2?"], max_new_tokens=16)
    print(f"Response: {responses2[0]}")

    phi.clear_overrides()
