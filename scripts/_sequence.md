in the experiment we run the scripts by this sequence:

1: alpaca_download
2: phi4mini_download
3: master_train
    └── imports from: activation_gather, sae_trainer, lfm_trainer
    └── for each layer 0-30:
        ├── activation_gather.gather_pair(layer)
        ├── sae_trainer.train_sae(mlp)
        ├── sae_trainer.train_sae(att)
        ├── lfm_trainer.train_lfm(layer)
        └── activation_gather.delete_pair(layer)
4: raw_dataset_activations_gather
5: feature_context_gather
6: autointerp_batch_submit
7: autointerp_batch_receive (can leave running in tmux, polls every 60s)
8: lfm_interp

optional:
- master_eval (runs all eval: test_activation_gather → sae_eval → lfm_eval)
  or run individually:
  - test_activation_gather (for eval data)
  - sae_eval (evaluate individual SAEs)
  - lfm_eval (evaluate LFM on test set)

structure:
- 62 SAEs: mlp[0-30] + att[1-31]
- 31 LFMs: mlp[N] → att[N+1] for N=0-30
- 62 interpretation files: mlp[0-30] + att[1-31]
