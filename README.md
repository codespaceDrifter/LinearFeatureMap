# Linear Feature Maps:interpret MLPs using linear layer to predict and map input and output SAE features

### goal:

to interpret MLP layers in transformers  

intermediate activations in transformers have been interpreted with sparse autoencoders by sprasely beaking them down into "features". see here:  
https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html  

short summary of SAE and features:  
we use a sparse autoencoder (SAE) which a  
linear (embed_dim, feature_dim), relu, linear (feature_dim, embed_dim)  
where the input and output is the SAME activation. the feature_dim is much larger than the embed_dim to extract common interpretable concepts. it is like grouping similar vectors and labeling them based on the token they are fron and the token they produced.    
to use an SAE, input a activation and see what absolutely horrible take. llms are no where near that level. maybe this is because my code is divine and unautomatable and your code is souless and reduceble to slop. eatures have the strongest values post relu. 

also through the attention score how strongly circuits attend to each other can be labeled. see here:  
https://transformer-circuits.pub/2025/attention-qk/index.html  

however there is no work that uses SAE features to study MLPs.  

anthropic did have the transcoder paper that encodes features for the input and outputs of a MLP plus attention layers. this is not an autoencoder because the inputs and outputs have some weight transformation in between them.  
https://transformer-circuits.pub/2025/attribution-graphs/biology.html  
i find this to be BAD.  
activations are states. they are a static concept.  
a weight is where those activations go. they are a function, a mapping, a transformation.   
therefore i do not think they should be interpreted together.  
they are defined only in relationship to each other (a activation is only defined based on the weights of the model they are in) but they should be interpreted as seperate things to really reach the smallest unit of interpretability.  
so ignoring transcoders as a bad direction let's interpret MLPs as a individual subject  

### thoughts:
so what is the funciton of MLPs (or kinda what some functions of intelligence). i think it is a transformation of concepts. there are two main ones: relationship mapping and causal mapping.  
relationship mapping means a "is a" relationship. it can go across abstraction levels (cat is mammal) or find attributes (sugar is tasty) or find similar concepts (planes to spaceships) or in terms of perception gradually assembling low level visuals into high level objects. 
 causal relationshpi mapping means a "leads to" relationship. this can be temporal (pushing button leads to elevator arriving), this can be logical (adding 3 to 2 equals 5)

### design:  

we train a linear regression model on the SAE features at the input of a MLP layer and the SAE features at the output of a MLP layer  
we gather activations based on the simpleQa dataset. maybe we do questions only and just let the models say whatever cause all we need is the activations. and the tokens they outputed.  
we label the features through examples with top activations  
the linear layer is of weight matrix shaped (feature_dim, feature_dim) and bias matrix shaped (feature dim)  

> $$\text{output} = \text{input} \cdot \text{Weight}^\text{T} + \text{Bias}$$

beyond mapping features we also want it to be sparse and be accurate to the actual activations, so we define the loss function as  

> $$ \text{loss} = (\text{f}_o - \text{LFM}(\text{f}_i))^2 + (\text{mlp}_{\text{out}} - \text{SAE}_{\text{decode}}(\text{LFM}(f_i)))^2 + \text{l1} $$

$f_i$ and $f_o$ are features in and features out, gotten by $\text{SAE}_{\text{encode}}(\text{mlp}_{\text{in}})$


after training we interpret the weights of the linear regression. let $W_{i,j}$ mean the value of the weight matrix row i col j.  
using the representation that each row means a neuron. the $i^{th}$ row is the row that dot products the entire in feature vector and outputs the predicted value of of the $i^{th}$ output feature. the $j^{th}$ column of each row is the weight that multiples the $j^{th}$ input feature.  
therefore, the $W_{i,j}$ is the weight that maps the $j^{th}$ input to the $i^{th}$ output.  
we then interpret the relationships between input and output features based on the value of the weight. a big weight would indicate a strong relationship and a very negative weight would indicate a inhibitory relationship. for this study we only study exitatory large value weights.  
we label the LFM as a map of the j column to the top k highest value i rows. so for each input feature what are the top k features it leads to as outputs.    
for example if i see the j feature is apple and the i feature is yummy and the $w_{i,j}$ is a big positive weight i can assume it associates apple with yummy.     
to intervention test we can modify the weights for example if original ffw maps "sleep" to "rest" we can change it to map "sleep" to "excitement" by negating the sleep weight from the rest neuron and adding a big value to the sleep weight to the excitement neuron. and we ask it i am tired what should i do maybe it will say "sit down and take a rest" rather than "go to sleep". or even crazier we can maybe do this to all ffws and see if we ask it "is sleeping a restful or exciting activity" and it says "exciting" tho maybe it won't work cause maybe this knowledge is stored elsewhere as well  


### implementaion:

we are using the phi4-mini-it model with 3.8b parameters  
we are going to print the architecture and inject SAEs with just the autocausalllm import  
we are going to train SAEs over the structured dataset Alphaca and we train only on the activation of the model generating (so not reading the question)  
we are training one LFW for each MLP so we gather activations pre and post each MLP    
this SAE is per layer pre and post MLP for LFW training.  
we will do 24576 for the SAE hidden_dim since the model embed_dim is 3072 embed_dim and we go for a 8x.  
we label SAE with the following series of structured information  
data/features/
  layer_8_pre/
    feature_0.json
    feature_1.json
    ...
  layer_8_post/
    ...

{
    top unembedding cosine similarity tokens: if we artifically put that activation to a high number and decode it and produce a embedding, what token embeddings is it most similar to

    a series of contexts based on activations above a certain threshold
    {
        output token index
        input question
        output answer
        causal connection: output answer with the activation artificially negated
    }
}

then we prompt a llm like claude opus 4.5 to interpret the features if possible. 

due to compute and time limitations we start with 4 layers of MLP not all layers: we pick layer 8, 16, 24, 31. 

for each we label the ones that fire > 0.5 only. many features will not be labeled.
we get a list of ids of all labeled features.

when training LFMs, we zero out ALL unlabeled features. 

for activations we are using the first 75% as training and last 25% as test. 


### potential problems:

note that a ffw is two matmuls with a relu in between. and maybe the smallest level of intereptability should be at the neuron level or should be at the single population level but for now we are at the double population level of the entire ffw as a unit of intereptability  

