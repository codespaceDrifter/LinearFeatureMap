in the experiment we run the scripts by this sequence:  

1: alpaca_download  
2: phi4mini_download  
3: activation_gather  
4: test_activation_gather  
5: sae_trainer
6: sae_eval
7: feature_data
8: restructure_feature_data
9: decoding_similarity
10: autointerp_batch_submit.py
11: autointerp_batch_receive.py
12: lfm_train.py
13: lfm_eval.py