from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
import torch
from inference import Phi4Inference

console = Console()
BLOCK_WIDTH = 12

def format_token(tok):
    return tok[:BLOCK_WIDTH-1].center(BLOCK_WIDTH-1)

def format_mag(mag):
    return f"{mag:^{BLOCK_WIDTH-1}.1f}"

def build_tables(tokens, activations, num_layers, blocks_per_row):
    """Build rich tables, one per wrap segment"""
    tables = []
    T = len(tokens)
    num_wraps = (T + blocks_per_row - 1) // blocks_per_row
    
    mags = activations.norm(dim=-1) if activations is not None else None  # (64, T)
    
    for wrap_idx in range(num_wraps):
        tok_start = wrap_idx * blocks_per_row
        tok_end = min(tok_start + blocks_per_row, T)
        wrap_tokens = tokens[tok_start:tok_end]
        
        table = Table(box=box.SIMPLE, padding=0, collapse_padding=True)
        
        # columns: label + tokens
        table.add_column("", style="bold", width=10)
        for i, t in enumerate(wrap_tokens):
            table.add_column(f"[{tok_start + i}] {format_token(t)}", width=BLOCK_WIDTH)
        
        # rows: L31 mlp, L31 att, L30 mlp, ...
        for layer_num in range(num_layers - 1, -1, -1):
            # mlp row
            mlp_idx = layer_num * 2 + 1
            row = [f"L{layer_num:02d} mlp"]
            for t in range(tok_start, tok_end):
                row.append(format_mag(mags[mlp_idx, t].item()) if mags is not None else "")
            table.add_row(*row)
            
            # att row
            att_idx = layer_num * 2
            row = [f"L{layer_num:02d} att"]
            for t in range(tok_start, tok_end):
                row.append(format_mag(mags[att_idx, t].item()) if mags is not None else "")
            table.add_row(*row)
        
        tables.append(table)
    
    return tables

def show_detail(tokens, activations, token_idx, layer_idx):
    """Detail panel for specific cell"""
    act = activations[layer_idx, token_idx]  # (3072,)
    layer_type = "att_in" if layer_idx % 2 == 0 else "mlp_in"
    layer_num = layer_idx // 2
    
    lines = [
        f"[bold]Layer {layer_num} {layer_type}[/bold] | Token {token_idx}: '{tokens[token_idx]}'",
        f"L2: {act.norm().item():.2f} | Mean: {act.mean().item():.4f} | Std: {act.std().item():.4f}",
        f"Min: {act.min().item():.4f} | Max: {act.max().item():.4f}",
        "",
        "[bold]Top 10 by magnitude:[/bold]"
    ]
    topk = act.abs().topk(10)
    for idx in topk.indices.tolist():
        lines.append(f"  [{idx:4d}]: {act[idx].item():+.4f}")
    
    return Panel("\n".join(lines), title="Detail", border_style="green")

def main():
    console.print("[bold]Loading model...[/bold]")
    phi = Phi4Inference()
    console.print("[green]Ready![/green]\n")
    
    while True:
        prompt = console.input("[bold cyan]>>> [/bold cyan]")
        
        if prompt.strip().lower() in ["quit", "exit", "q"]:
            break
        if not prompt.strip():
            continue
        
        console.print("[dim]Generating...[/dim]")
        responses, tokens, acts = phi.generate([prompt], max_new_tokens=64)
        
        response = responses[0]
        toks = tokens[0]  # [str]
        activations = acts[0]  # (64, T, 3072)
        
        console.clear()
        console.print(f"[bold]Prompt:[/bold] {prompt}")
        console.print(f"[bold]Response:[/bold] {response}\n")
        
        blocks_per_row = max(1, (console.width - 12) // BLOCK_WIDTH)
        tables = build_tables(toks, activations, phi.num_layers, blocks_per_row)
        for table in tables:
            console.print(table)
            console.print()
        
        console.print("[dim]Enter 'token_idx,layer_idx' to inspect (e.g. '3,17'), Enter to continue[/dim]")
        while True:
            cmd = console.input("[yellow]inspect> [/yellow]")
            if not cmd.strip():
                break
            try:
                token_idx, layer_idx = map(int, cmd.split(","))
                if 0 <= token_idx < len(toks) and 0 <= layer_idx < phi.num_layers * 2:
                    console.print(show_detail(toks, activations, token_idx, layer_idx))
                else:
                    console.print("[red]Out of range[/red]")
            except:
                console.print("[red]Format: token_idx,layer_idx[/red]")
    
    console.print("[bold]Bye![/bold]")

if __name__ == "__main__":
    main()
