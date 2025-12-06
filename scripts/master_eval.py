"""
Master evaluation script.
Runs: test_activation_gather → sae_eval → lfm_eval
"""
from scripts.config import config, pathconfig
from scripts.test_activation_gather import gather_test_activations
from scripts.sae_eval import eval_sae
from scripts.lfm_eval import eval_layer
import os

num_layers = config["num_layers"]


def main():
    print("=" * 60)
    print("STEP 1: Gathering test activations")
    print("=" * 60)

    # check if already gathered
    if os.path.exists(pathconfig["test_metadata"]):
        print("Test activations already exist, skipping...")
    else:
        gather_test_activations()

    print("\n" + "=" * 60)
    print("STEP 2: Evaluating SAEs")
    print("=" * 60)

    # eval mlp SAEs (layers 0 to 30)
    for layer in range(num_layers - 1):
        eval_sae("mlp", layer)

    # eval att SAEs (layers 1 to 31)
    for layer in range(1, num_layers):
        eval_sae("att", layer)

    print("\n" + "=" * 60)
    print("STEP 3: Evaluating LFMs")
    print("=" * 60)

    # eval LFMs (layers 0 to 30)
    for layer in range(num_layers - 1):
        eval_layer(layer)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
