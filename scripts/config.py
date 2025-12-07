# Common configuration values used across multiple scripts
#
# TODO: eval_split is temporarily set to 0.95-1.0 (5%) for faster eval
# Change back to 0.75-1.0 (25%) for full eval later
#
config = {
    # Model architecture
    "num_layers": 32,  # total transformer layers
    "embed_dim": 3072,
    "hidden_dim": 12288,  # embed_dim * 4

    # Device
    "device": "cuda",

    # Generation
    "max_new_tokens": 64,

    # Data split
    "split": (0, 0.75),  # train
    "eval_split": (0.95, 1.0),  # eval (TODO: change to 0.75-1.0 for full eval)
}


# Path configuration with clean accessors
class PathConfig:
    """
    Usage:
        # Weights - separate SAEs for mlp_in and att_in
        pathconfig["sae"]["mlp"][layer]       -> "./weights/SAE/layer_{layer}_mlp_sae.pt"
        pathconfig["sae"]["att"][layer]       -> "./weights/SAE/layer_{layer}_att_sae.pt"
        pathconfig["lfm"][layer]              -> "./weights/LFM/layer_{layer}_lfm.pt"
        pathconfig["model"]                   -> "./weights/phi4-mini"

        # Raw activations (binary) - NO implicit layer+1, explicit positions
        pathconfig["activations"]["mlp"][layer]      -> "./data/activations/layer_{layer}_mlp_in.bin"
        pathconfig["activations"]["att"][layer]      -> "./data/activations/layer_{layer}_att_in.bin"
        pathconfig["test_activations"]["mlp"][layer] -> "./data/test/activations/..."
        pathconfig["test_activations"]["att"][layer] -> "./data/test/activations/..."
        pathconfig["metadata"]                       -> "./data/activations/metadata.npy"
        pathconfig["test_metadata"]                  -> "./data/test/activations/metadata.npy"

        # Contexts folder (preparation/hydration data)
        pathconfig["raw_activations"]["mlp"][layer]  -> "./data/contexts/layer_{layer}_mlp_raw.jsonl"
        pathconfig["raw_activations"]["att"][layer]  -> "./data/contexts/layer_{layer}_att_raw.jsonl"
        pathconfig["feature_context"]["mlp"][layer]  -> "./data/contexts/layer_{layer}_mlp_context.json"
        pathconfig["feature_context"]["att"][layer]  -> "./data/contexts/layer_{layer}_att_context.json"
        pathconfig["example_hydrate"]                -> "./data/contexts/example_hydrate.jsonl"

        # Features folder (final interpreted features)
        pathconfig["interpretations"]["mlp"][layer]  -> "./data/features/layer_{layer}_mlp.json"
        pathconfig["interpretations"]["att"][layer]  -> "./data/features/layer_{layer}_att.json"
        pathconfig["batch_ids"]                      -> "./data/features/batch_ids.txt"

        # LFM labels
        pathconfig["lfm_labels"][layer]              -> "./data/labels/lfm_{layer}.json"

        # Data sources
        pathconfig["alpaca"]                         -> "./data/alpaca"

    LFM for layer N maps: sae["mlp"][N] features -> sae["att"][N+1] features
    (layer N mlp_in -> layer N+1 att_in, i.e., across the MLP)
    """

    def __getitem__(self, key):
        if key == "sae":
            return _SAEPaths()
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
        elif key == "lfm_labels":
            return _LFMLabelPaths()
        elif key == "metadata":
            return "./data/activations/metadata.npy"
        elif key == "test_metadata":
            return "./data/test/activations/metadata.npy"
        elif key == "model":
            return "./weights/phi4-mini"
        elif key == "alpaca":
            return "./data/alpaca"
        elif key == "merged_sae":
            return "./weights/SAE/merged_sae.pt"
        else:
            raise KeyError(f"Unknown path key: {key}")


class _SAEPaths:
    """pathconfig["sae"]["mlp"][layer] or pathconfig["sae"]["att"][layer]"""
    def __getitem__(self, kind):
        if kind == "mlp":
            return _SAEKindPaths("mlp")
        elif kind == "att":
            return _SAEKindPaths("att")
        else:
            raise KeyError(f"Unknown SAE kind: {kind}. Use 'mlp' or 'att'")


class _SAEKindPaths:
    def __init__(self, kind):
        self.kind = kind

    def __getitem__(self, layer):
        return f"./weights/SAE/layer_{layer}_{self.kind}_sae.pt"


class _LFMPaths:
    def __getitem__(self, layer):
        return f"./weights/LFM/layer_{layer}_lfm.pt"


class _LFMLabelPaths:
    def __getitem__(self, layer):
        return f"./data/labels/lfm_{layer}.json"


class _ActivationPaths:
    """pathconfig["activations"]["mlp"][layer] or pathconfig["activations"]["att"][layer]"""
    def __init__(self, test=False):
        self.base = "./data/test/activations" if test else "./data/activations"

    def __getitem__(self, kind):
        if kind == "mlp":
            return _ActivationKindPaths(self.base, "mlp")
        elif kind == "att":
            return _ActivationKindPaths(self.base, "att")
        else:
            raise KeyError(f"Unknown activation kind: {kind}. Use 'mlp' or 'att'")


class _ActivationKindPaths:
    def __init__(self, base, kind):
        self.base = base
        self.kind = kind

    def __getitem__(self, layer):
        return f"{self.base}/layer_{layer}_{self.kind}_in.bin"


class _RawActivationPaths:
    """pathconfig["raw_activations"]["mlp"][layer] or pathconfig["raw_activations"]["att"][layer]"""
    def __getitem__(self, kind):
        if kind == "mlp":
            return _RawActivationKindPaths("mlp")
        elif kind == "att":
            return _RawActivationKindPaths("att")
        else:
            raise KeyError(f"Unknown kind: {kind}. Use 'mlp' or 'att'")


class _RawActivationKindPaths:
    def __init__(self, kind):
        self.kind = kind

    def __getitem__(self, layer):
        return f"./data/contexts/layer_{layer}_{self.kind}_raw.jsonl"


class _FeatureContextPaths:
    """pathconfig["feature_context"]["mlp"][layer] or pathconfig["feature_context"]["att"][layer]"""
    def __getitem__(self, kind):
        if kind == "mlp":
            return _FeatureContextKindPaths("mlp")
        elif kind == "att":
            return _FeatureContextKindPaths("att")
        else:
            raise KeyError(f"Unknown kind: {kind}. Use 'mlp' or 'att'")


class _FeatureContextKindPaths:
    def __init__(self, kind):
        self.kind = kind

    def __getitem__(self, layer):
        return f"./data/contexts/layer_{layer}_{self.kind}_context.json"


class _InterpretationPaths:
    """pathconfig["interpretations"]["mlp"][layer] or pathconfig["interpretations"]["att"][layer]"""
    def __getitem__(self, kind):
        if kind == "mlp":
            return _InterpretationKindPaths("mlp")
        elif kind == "att":
            return _InterpretationKindPaths("att")
        else:
            raise KeyError(f"Unknown kind: {kind}. Use 'mlp' or 'att'")


class _InterpretationKindPaths:
    def __init__(self, kind):
        self.kind = kind

    def __getitem__(self, layer):
        return f"./data/features/layer_{layer}_{self.kind}.json"


pathconfig = PathConfig()
