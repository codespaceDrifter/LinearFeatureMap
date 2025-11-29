import torch
import json
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from phi4_mini.inference import Phi4Inference
from SAE.SAE import SAE

LAYERS = [8, 16, 24]
EMBED_DIM = 3072
HIDDEN_DIM = 12288
DEVICE = "cuda"
TOP_K = 10

phi = Phi4Inference(layers=LAYERS, device=DEVICE)
token_embeddings = phi.model.model.embed_tokens.weight.detach()  # (vocab_size, 3072)
embed_norm = token_embeddings / token_embeddings.norm(dim=1, keepdim=True)

for layer in LAYERS:
    for pos in ["pre", "post"]:
        print(f"Processing layer_{layer}_{pos}...")
        
        sae = SAE(EMBED_DIM, HIDDEN_DIM).to(DEVICE)
        sae.load_state_dict(torch.load(f"./weights/SAE/layer_{layer}_{pos}_sae.pt"))
        sae.eval()
        
        decoder = sae.decoder.weight.detach()  # (embed_dim, hidden_dim) or (hidden_dim, embed_dim)?
        
        with open(f"./data/features/layer_{layer}_{pos}.json", "r") as f:
            features = json.load(f)
        
        for fid in features:
            feature_vector = decoder[:, int(fid)]  # (3072,) - adjust indexing if needed
            feature_norm = feature_vector / feature_vector.norm()
            
            similarities = embed_norm.float() @ feature_norm  # (vocab_size,)
            top_k = similarities.topk(TOP_K)
            
            top_tokens = [phi.tokenizer.decode([i]) for i in top_k.indices.tolist()]
            top_scores = [round(s, 3) for s in top_k.values.tolist()]
            
            features[fid]["decoding"] = {"tokens": top_tokens, "scores": top_scores}
        
        with open(f"./data/features/layer_{layer}_{pos}.json", "w") as f:
            json.dump(features, f)
        
        print(f"  {len(features)} features updated")

print("Done!")