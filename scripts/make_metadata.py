import numpy as np
import os

# check actual file size
file_size = os.path.getsize("./data/activations/layer_8_pre.bin")
embed_dim = 3072
total_tokens = file_size // (4 * embed_dim)  # 4 bytes per float32

print(f"Total tokens: {total_tokens}")

np.save("./data/activations/metadata.npy", {
    "total_tokens": total_tokens,
    "embed_dim": 3072,
    "layers": [8, 16, 24, 31],
    "dtype": "float32"
})