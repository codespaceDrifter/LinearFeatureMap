# Common configuration values used across multiple scripts
config = {
    # Model architecture
    "layers": [8, 16, 24],
    "embed_dim": 3072,
    "hidden_dim": 12288,  # embed_dim * 4

    # Device
    "device": "cuda",

    # Generation
    "max_new_tokens": 64,

    # Data split (train)
    "split": (0, 0.75),
}


# Path configuration with clean accessors
class PathConfig:
    """
    Usage:
        # Weights
        pathconfig["sae"][layer]              -> "./weights/SAE/layer_{layer}_sae.pt"
        pathconfig["merged_sae"]              -> "./weights/SAE/merged_sae.pt"
        pathconfig["lfm"][layer]              -> "./weights/LFM/layer_{layer}_lfm.pt"
        pathconfig["model"]                   -> "./weights/phi4-mini"

        # Raw activations (binary)
        pathconfig["activations"][layer]["mlp"]      -> "./data/activations/layer_{layer}_mlp_in.bin"
        pathconfig["activations"][layer]["att"]      -> "./data/activations/layer_{layer+1}_att_in.bin"
        pathconfig["test_activations"][layer]["mlp"] -> "./data/test/activations/..."
        pathconfig["test_activations"][layer]["att"] -> "./data/test/activations/..."
        pathconfig["metadata"]                       -> "./data/activations/metadata.npy"
        pathconfig["test_metadata"]                  -> "./data/test/activations/metadata.npy"

        # Contexts folder (preparation/hydration data)
        pathconfig["raw_activations"][layer]  -> "./data/contexts/raw_dataset_activations_layer_{layer}.jsonl"
        pathconfig["feature_context"][layer]  -> "./data/contexts/feature_context_layer_{layer}.json"
        pathconfig["example_hydrate"]         -> "./data/contexts/example_hydrate.jsonl"

        # Features folder (final interpreted features)
        pathconfig["interpretations"][layer]  -> "./data/features/layer_{layer}.json"
        pathconfig["batch_ids"]               -> "./data/features/batch_ids.txt"

        # Data sources
        pathconfig["alpaca"]                  -> "./data/alpaca"
    """

    def __getitem__(self, key):
        if key == "sae":
            return _SAEPaths()
        elif key == "merged_sae":
            return "./weights/SAE/merged_sae.pt"
        elif key == "lfm":
            return _LFMPaths()
        elif key == "activations":
            return _ActivationPaths(test=False)
        elif key == "test_activations":
            return _ActivationPaths(test=True)
        elif key == "raw_activations":
            return _RawActivationPaths()
        elif key == "feature_context":
            return _FeatureContextPaths()
        elif key == "example_hydrate":
            return "./data/contexts/example_hydrate.jsonl"
        elif key == "interpretations":
            return _InterpretationPaths()
        elif key == "batch_ids":
            return "./data/features/batch_ids.txt"
        elif key == "metadata":
            return "./data/activations/metadata.npy"
        elif key == "test_metadata":
            return "./data/test/activations/metadata.npy"
        elif key == "model":
            return "./weights/phi4-mini"
        elif key == "alpaca":
            return "./data/alpaca"
        else:
            raise KeyError(f"Unknown path key: {key}")


class _SAEPaths:
    def __getitem__(self, layer):
        return f"./weights/SAE/layer_{layer}_sae.pt"


class _LFMPaths:
    def __getitem__(self, layer):
        return f"./weights/LFM/layer_{layer}_lfm.pt"


class _ActivationPaths:
    def __init__(self, test=False):
        self.base = "./data/test/activations" if test else "./data/activations"

    def __getitem__(self, layer):
        return _ActivationLayerPaths(self.base, layer)


class _ActivationLayerPaths:
    def __init__(self, base, layer):
        self.base = base
        self.layer = layer

    def __getitem__(self, kind):
        if kind == "mlp":
            return f"{self.base}/layer_{self.layer}_mlp_in.bin"
        elif kind == "att":
            return f"{self.base}/layer_{self.layer + 1}_att_in.bin"
        else:
            raise KeyError(f"Unknown activation kind: {kind}. Use 'mlp' or 'att'")


class _RawActivationPaths:
    def __getitem__(self, layer):
        return f"./data/contexts/raw_dataset_activations_layer_{layer}.jsonl"


class _FeatureContextPaths:
    def __getitem__(self, layer):
        return f"./data/contexts/feature_context_layer_{layer}.json"


class _InterpretationPaths:
    def __getitem__(self, layer):
        return f"./data/features/layer_{layer}.json"


pathconfig = PathConfig()
