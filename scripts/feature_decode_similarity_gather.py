import torch
import json
from phi4_mini.inference import Phi4Inference
from interp_algo.SAE import SAE
from scripts.config import config, pathconfig

TOP_K = 4  # top decoded tokens to keep

phi = Phi4Inference(device=config["device"])
token_embeddings = phi.model.model.embed_tokens.weight.detach()  # (vocab_size, 3072)
embed_norm = token_embeddings / token_embeddings.norm(dim=1, keepdim=True)

for layer in config["layers"]:
    print(f"Processing layer {layer}...")

    sae = SAE(config["embed_dim"], config["hidden_dim"]).to(config["device"])
    sae.load_state_dict(torch.load(pathconfig["sae"][layer]))
    sae.eval()

    decoder = sae.decoder.weight.detach()  # (embed_dim, hidden_dim)

    with open(pathconfig["feature_context"][layer], "r") as f:
        features = json.load(f)
    
    for fid in features:
        feature_vector = decoder[:, int(fid)]  # (3072,)
        feature_norm = feature_vector / feature_vector.norm()
        
        similarities = embed_norm.float() @ feature_norm  # (vocab_size,)
        top_k = similarities.topk(TOP_K)
        
        top_tokens = [phi.tokenizer.decode([i]) for i in top_k.indices.tolist()]
        top_scores = [round(s, 2) for s in top_k.values.tolist()]
        
        features[fid]["decoding"] = {"tokens": top_tokens, "scores": top_scores}
    
    with open(pathconfig["feature_context"][layer], "w") as f:
        json.dump(features, f)
    
    print(f"  {len(features)} features updated")

print("Done!")