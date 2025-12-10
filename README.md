SAEs for phi4mini and Linear Feature Maps to study MLPs as linear regression for feature input and output. 

future changes.  
train on fineweb-edu 10b tokens. train on inputs instead of output. since the model is still thinking and trying to predict when reading and refining it's context stream. stream activations no storing. chunked random sampling each chunk 1024 tokens 32 chunks. load a batch of chunked continuous tokens and train SAE. do the same for LFM.  
then we label by sequential chunking and dividing of inputs (no output) and token activations