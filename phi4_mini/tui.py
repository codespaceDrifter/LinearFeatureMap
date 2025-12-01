import sys
import torch
from inference import Phi4Inference

# ANSI color codes for heatmap
def activation_color(val, max_val):
    """Map activation magnitude to color intensity"""
    if max_val == 0:
        return "\033[48;5;232m"  # dark gray
    ratio = min(val / max_val, 1.0)
    # 232-255 are grayscale, 196-226 are reds/yellows
    if ratio < 0.3:
        code = 232 + int(ratio * 30)  # dark grays
    elif ratio < 0.6:
        code = 58 + int((ratio - 0.3) * 20)  # yellows
    else:
        code = 196 + int((ratio - 0.6) * 15)  # reds
    return f"\033[48;5;{code}m"

def reset():
    return "\033[0m"

def clear_screen():
    print("\033[2J\033[H", end="")

def visualize(tokens, activations, num_layers=32):
    """
    tokens: list of token strings, length T
    activations: (64, T, 3072) tensor - 64 = num_layers * 2 (att_in, mlp_in alternating)
    """
    T = len(tokens)
    if T == 0:
        print("No tokens generated")
        return
    
    # compute activation magnitudes per layer per token: (64, T)
    mags = activations.norm(dim=-1)  # L2 norm across embed dim
    max_mag = mags.max().item()
    
    # truncate tokens for display
    max_tok_len = 12
    display_tokens = [t[:max_tok_len].ljust(max_tok_len) for t in tokens]
    
    # header
    print("\n" + " " * 14 + "".join(display_tokens))
    print(" " * 14 + "-" * (max_tok_len * T))
    
    # print layers from top (layer 31) to bottom (layer 0)
    for l in range(num_layers - 1, -1, -1):
        # mlp_in row
        row = f"L{l:02d} mlp_in  |"
        for t in range(T):
            mag = mags[l * 2 + 1, t].item()
            color = activation_color(mag, max_mag)
            row += f"{color}{mag:11.1f} {reset()}"
        print(row)
        
        # att_in row
        row = f"L{l:02d} att_in  |"
        for t in range(T):
            mag = mags[l * 2, t].item()
            color = activation_color(mag, max_mag)
            row += f"{color}{mag:11.1f} {reset()}"
        print(row)
        
        if l > 0:
            print(" " * 14 + "·" * (max_tok_len * T))
    
    print()

def main():
    print("Loading model...")
    phi = Phi4Inference()
    print("Ready!\n")
    
    while True:
        try:
            prompt = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        
        if not prompt.strip():
            continue
        
        if prompt.strip().lower() in ["quit", "exit", "q"]:
            break
        
        print("Generating...")
        responses, tokens, acts = phi.generate([prompt], max_new_tokens=32)
        
        clear_screen()
        print(f"Prompt: {prompt}")
        print(f"Response: {responses[0]}")
        
        visualize(tokens[0], acts[0], num_layers=phi.num_layers)

if __name__ == "__main__":
    main()