in the experiment we run the scripts by this sequence:

1: alpaca_download
2: phi4mini_download
3: master_train
    └── for each layer pair (processes 64 SAEs + 31 LFMs):
        ├── gather activations (mlp_in[N] + att_in[N+1])
        ├── train SAE for mlp_in[N]
        ├── train SAE for att_in[N+1]
        ├── train LFM: mlp_in[N] → att_in[N+1]
        └── delete activations
4: raw_dataset_activations_gather
5: feature_context_gather
6: autointerp_batch_submit
7: autointerp_batch_receive (can leave running in tmux, polls every 60s)
8: lfm_interp

optional:
- test_activation_gather (for eval data)
- lfm_eval (evaluate LFM on test set)
- sae_eval (evaluate individual SAEs)

structure:
- 64 SAEs: mlp[0-31] + att[0-31]
- 31 LFMs: mlp[N] → att[N+1] for N=0-30
- 64 interpretation files: mlp[0-31] + att[0-31]
