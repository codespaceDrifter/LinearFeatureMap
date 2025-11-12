# Linear Feature Maps:interpret MLPs using linear layer to predict and map input and output SAE features

### goal:

intermediate activations in transformers have been interpreted with sparse autoencoders by sprasely beaking them down into "features". see here:  
https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html  

short summary of SAE and features:  
we use a sparse autoencoder (SAE) which a  
linear (embed_dim, feature_dim), relu, linear (feature_dim, embed_dim)  
where the input and output is the SAME activation. the feature_dim is much larger than the embed_dim to extract common interpretable concepts. it is like grouping similar vectors and labeling them based on the token they are fron and the token they produced.    
to use an SAE, input a activation and see what features have the strongest values post relu. 

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
so what is the funciton of MLPs (or kinda what some functions of intelligence). i think there are two main ones: relationship mapping and causal mapping.  
relationship mapping means a "is a" relationship. it can go across abstraction levels (cat is mammal) or find attributes (sugar is tasty) or find similar concepts (planes to spaceships) or in terms of perception gradually assembling low level visuals into high level objects. 
 causal relationshpi mapping means a "leads to" relationship. this can be temporal (pushing button leads to elevator arriving), this can be logical (adding 3 to 2 equals 5)

### design:  

we train a linear regression model on the SAE features at the input of a MLP layer and the SAE features at the output of a MLP layer  
the linear layer is of weight matrix shaped (feature_dim, feature_dim) and bias matrix shaped (feature dim)  

> $$output = input \cdot Weight.T  + Bias$$

beyond mapping features we also want it to be sparse and be accurate to the actual activations, so we define the loss function as  

> $$ loss = (f_o - LFM(f_i))^2 + (mlp_{out} - SAE_{decode}(LFM(SAE(mlp_{in}))))^2 + l1$$

after training we interpret the weights of the linear regression. let $W_{i,j}$ mean the value of the weight matrix row i col j.  
using the representation that each row means a neuron. the $i^{th}$ row is the row that dot products the entire in feature vector and outputs the predicted value of of the $i^{th}$ output feature. the $j^{th}$ column of each row is the weight that multiples the $j^{th}$ input feature.  
therefore, the $W_{i,j}$ is the weight that maps the $j^{th}$ input to the $i^{th}$ output.  
we then interpret the relationships between input and output features based on the value of the weight. a big weight would indicate a strong relationship and a very negative weight would indicate a inhibitory relationship. for this study we only study exitatory large value weights.  
we train one SAE on all intermediate actiavtions and one LFM on all MLPs. i can see how this cause problems, discussed later in the problems section.    
we label the SAE features according to the tokens that produced it on the first layer and the tokens it leads the model to output on the last layer. we label the LFM as a map of the j column to the top k highest value i rows.   
for example if i see the j feature is apple and the i feature is yummy and the $w_{i,j}$ is a big positive weight i can assume it associates apple with yummy.     
when analyzing specific tokens in a actual example, we label LFM as a map between the top k input features and the top k output features.  
to intervention test we can modify the weights for example if original ffw maps "sleep" to "rest" we can change it to map "sleep" to "excitement" by negating the sleep weight from the rest neuron and adding a big value to the sleep weight to the excitement neuron. and we ask it i am tired what should i do maybe it will say "sit down and take a rest" rather than "go to sleep". or even crazier we can maybe do this to all ffws and see if we ask it "is sleeping a restful or exciting activity" and it says "exciting" tho maybe it won't work cause maybe this knowledge is stored elsewhere as well  


### implementaion:

we are using the gemma3-1b model as the model and the simpleQA dataset as the dataset to train the SAE and the FM. 
train a common SaE on activations, label them as features by automatically prompting a advanced model like claude.  

### potential problems:
for now we train a single SFM across the input outputs features of all MLP layers. note that this could be a problem and also training a single SaE on all activations could be a problem. because activations are defined based on the neurons they are connected to. and they are connected to different neurons depending on what layers they are at so the similar activations on different layers can mean very different things. also different MLPs could be very different maps so training a single SFM could be a problem. maybe we can train a seperate SaE and SFM for each activation and each MLP but the labeling is a problem without a direct text input and output to label from. 

note that a ffw is two matmuls with a relu in between. and maybe the smallest level of intereptability should be at the neuron level or should be at the single population level but for now we are at the double population level of the entire ffw as a unit of intereptability  
