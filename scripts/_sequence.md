in the experiment we run the scripts by this sequence:

1: alpaca_download
2: phi4mini_download
3: master_train
    └── imports from: activation_gather, sae_trainer, lfm_trainer
    └── processes in layer batches of 8 (0-7, 8-15, 16-23, 24-30):
        ├── activation_gather.gather_layer_batch(layers) - one pass, saves all 8 layers
        ├── for each layer in batch:
        │   ├── sae_trainer.train_sae(mlp)
        │   ├── sae_trainer.train_sae(att)
        │   └── lfm_trainer.train_lfm(layer)
        └── activation_gather.delete_layer_batch(layers)
4: raw_dataset_activations_gather
5: feature_context_gather
6: autointerp_batch_submit
7: autointerp_batch_receive (can leave running in tmux, polls every 60s)
8: create_merged_sae (creates merged SAE for TUI from individual SAEs)
9: lfm_interp

optional:
- master_eval (same structure as master_train but for test split)
    └── processes in layer batches of 8:
        ├── test_activation_gather.gather_test_layer_batch(layers)
        ├── for each layer in batch:
        │   ├── sae_eval.eval_sae(mlp)
        │   ├── sae_eval.eval_sae(att)
        │   └── lfm_eval.eval_lfm(layer)
        └── test_activation_gather.delete_test_layer_batch(layers)

structure:
- 62 SAEs: mlp[0-30] + att[1-31]
- 31 LFMs: mlp[N] → att[N+1] for N=0-30
- 62 interpretation files: mlp[0-30] + att[1-31]
