import json
import copy
import anthropic

client = anthropic.Anthropic()

# load examples
examples = {}
with open("./data/examples.jsonl", "r") as f:
    for line in f:
        ex = json.loads(line)
        examples[ex["id"]] = {"input": ex["input"], "output": ex["output"]}

SYSTEM_PROMPT = """
Hi claudy! You will help me label features for mechanistic interpretability of Sparse Autoencoder.
You'll see:
1. Contexts where this feature activated (we trained on model outputs in the alpaca dataset) Context includes:
Activations. which are the tokens in the answer that led this feature to fire. formated as [index] , 'token' , activation value
the input question in the dataset
the output answer in the model wrote

2. Top decoded tokens: top 10 dot product of decoder weights in SAE for that feature with all token embeddings (both normalized)
formatted as first the list of top decoded tokens and then their corresponding scores

Your job:
first, critically think a bit with your thinking tokens and determine what this feature means. consider all provided information do not be lazy and just look at one and pick that. 
make sure the feature interpretation you draft actually explains many or most of the contexts pretty well. top decoded could provide clues but the middle layers might not be actually 
embedding space so maybe they won't make sense. also the context can provide clues but maybe it just numerically fits that it fires without a strong semantic meaning.
You're smart. make your best decision.  
now you are done thinking you can write your label. make sure it is CONCISE AND CLEAR within 10 words or so. Optionally after your label if this feature requires more explanation, you can
put a colon : and then provide more explanation. 
If no clear pattern exists, which many features might have, do not force one, just say:
UNINTERPRETABLE somewhere in your output answer(just the word uninterpetable case doesn't matter). and i will programmatically ignore them for later use.
Output only your label, nothing else. no courtesy or 'ok let me start now' or reasoning steps keep them in thinking tokens. 
because i will directly display and read your output as the labels"""

def get_text(response):
    for block in response.content:
        if block.type == "text":
            return block.text.strip()
    return ""

def call_with_thinking(msg, budget):
    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=budget + 200,
        thinking={"type": "enabled", "budget_tokens": budget},
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": msg}]
    )
    return get_text(response)

def call_no_thinking(msg):
    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=200,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": msg}]
    )
    return get_text(response)

# iterate through features
with open("./data/features/layer_8_pre.json", "r") as f:
    features = json.load(f)

for fid, fdata in features.items():
    hydrated = copy.deepcopy(fdata)
    
    for ctx in hydrated["contexts"]:
        ex = examples.get(ctx["example_id"])
        if ex:
            ctx["input"] = ex["input"][:300]
            ctx["output"] = ex["output"][:300]
            del ctx["example_id"]
    
    msg = f"Feature {fid} | layer 8 pre-MLP\n\n{json.dumps(hydrated, indent=2)}"
    
    print(f"\n{'='*60}")
    print(f"Feature {fid}")
    print(f"{'='*60}")
    
    label_2000 = call_with_thinking(msg, 2000)
    print(f"2000 thinking: {label_2000}")
    
    label_1000 = call_with_thinking(msg, 1000)
    print(f"1000 thinking: {label_1000}")
    
    label_none = call_no_thinking(msg)
    print(f"no thinking:   {label_none}")
    
    input("\n[Enter to continue]")
