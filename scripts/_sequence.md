in the experiment we run the scripts by this sequence:

1: alpaca_download
2: phi4mini_download
3: activation_gather
4: test_activation_gather
5: sae_trainer
6: sae_eval
7: create_merged_sae
8: raw_dataset_activations_gather
9: feature_context_gather
10: feature_decode_similarity_gather
11: autointerp_batch_submit
12: autointerp_batch_receive
13: lfm_trainer
14: lfm_eval
