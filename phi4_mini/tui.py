import json
import os
import sys

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phi4_mini.feature_inference import Phi4FeatureInference
from scripts.config import config, pathconfig

console = Console()
BLOCK_WIDTH = 12
FEATURE_THRESHOLD = 0.75


def load_labels():
    """Load interpretation labels for all SAEs"""
    labels = {"mlp": {}, "att": {}}
    num_layers = config["num_layers"]

    for layer in range(num_layers - 1):  # 0 to 30
        path = pathconfig["interpretations"]["mlp"][layer]
        if os.path.exists(path):
            with open(path, "r") as f:
                labels["mlp"][layer] = json.load(f)

    for layer in range(1, num_layers):  # 1 to 31
        path = pathconfig["interpretations"]["att"][layer]
        if os.path.exists(path):
            with open(path, "r") as f:
                labels["att"][layer] = json.load(f)

    console.print(f"[dim]Loaded {len(labels['mlp'])} mlp + {len(labels['att'])} att label files[/dim]")
    return labels


def get_firing_features(features, hook_idx, token_idx):
    """Get features firing above threshold for a specific cell."""
    if features is None:
        return []

    z = features[hook_idx, token_idx]  # (12288,)
    mask = z > FEATURE_THRESHOLD
    if not mask.any():
        return []

    indices = mask.nonzero().squeeze(-1).tolist()
    if isinstance(indices, int):
        indices = [indices]

    return [(idx, z[idx].item()) for idx in indices]


def format_token(tok):
    return tok[:BLOCK_WIDTH-1].center(BLOCK_WIDTH-1)


def format_cell(num_features):
    """Format cell: feature count"""
    if num_features > 0:
        return f"[green]{num_features}f[/green]"
    return "[dim]0[/dim]"


def build_tables(tokens, features, num_layers, blocks_per_row):
    """Build rich tables, one per wrap segment"""
    tables = []
    T = len(tokens)
    num_wraps = (T + blocks_per_row - 1) // blocks_per_row

    # precompute firing feature counts for all cells
    feature_counts = {}  # (hook_idx, token_idx) -> count
    if features is not None:
        fired = (features > FEATURE_THRESHOLD).sum(dim=-1)  # (64, T)
        for hook_idx in range(num_layers * 2):
            for t in range(T):
                feature_counts[(hook_idx, t)] = fired[hook_idx, t].item()

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
                count = feature_counts.get((mlp_idx, t), 0)
                row.append(format_cell(count))
            table.add_row(*row)

            # att row
            att_idx = layer_num * 2
            row = [f"L{layer_num:02d} att"]
            for t in range(tok_start, tok_end):
                count = feature_counts.get((att_idx, t), 0)
                row.append(format_cell(count))
            table.add_row(*row)

        tables.append(table)

    return tables


def show_detail(tokens, features, token_idx, layer_idx, labels):
    """Detail panel for specific cell with feature labels"""
    layer_type = "att" if layer_idx % 2 == 0 else "mlp"
    layer_num = layer_idx // 2

    lines = [
        f"[bold]Layer {layer_num} {layer_type}_in[/bold] | Token {token_idx}: '{tokens[token_idx]}'",
        "",
    ]

    # get firing features with labels
    layer_labels = labels.get(layer_type, {}).get(layer_num, {})

    if features is not None:
        fires = get_firing_features(features, layer_idx, token_idx)
        fires.sort(key=lambda x: -x[1])  # sort by activation value

        if fires:
            lines.append(f"[bold green]Firing features (>{FEATURE_THRESHOLD}):[/bold green]")
            for fid, fval in fires[:20]:  # show top 20
                label = layer_labels.get(str(fid), {}).get("interpretation", "[no label]")
                lines.append(f"  [yellow]f{fid}[/yellow] ({fval:.2f}): {label}")
            if len(fires) > 20:
                lines.append(f"  ... and {len(fires) - 20} more")
        else:
            lines.append("[dim]No features firing above threshold[/dim]")
    else:
        lines.append("[dim]No features available[/dim]")

    return Panel("\n".join(lines), title="Detail", border_style="green")


def main():
    console.print("[bold]Loading Phi4 + SAEs...[/bold]")
    try:
        phi = Phi4FeatureInference()
    except AssertionError as e:
        console.print(f"[red]Error loading SAEs:[/red]\n{e}")
        console.print("[dim]Run: python -m scripts.master_train[/dim]")
        return

    labels = load_labels()
    console.print("[green]Ready![/green]\n")

    while True:
        prompt = console.input("[bold cyan]>>> [/bold cyan]")

        if prompt.strip().lower() in ["quit", "exit", "q"]:
            break
        if not prompt.strip():
            continue

        console.print("[dim]Generating...[/dim]")
        responses, tokens, features = phi.generate([prompt], max_new_tokens=64)

        response = responses[0]
        toks = tokens[0]  # [str]
        feat = features[0]  # (64, T, 12288)

        console.clear()
        console.print(f"[bold]Prompt:[/bold] {prompt}")
        console.print(f"[bold]Response:[/bold] {response}\n")

        blocks_per_row = max(1, (console.width - 12) // BLOCK_WIDTH)
        tables = build_tables(toks, feat, phi.num_layers, blocks_per_row)
        for table in tables:
            console.print(table)
            console.print()

        console.print("[dim]Cells show: [green]Nf[/green] = N features firing[/dim]")
        console.print("[dim]Enter 'token_idx,layer_idx' to inspect (e.g. '3,17'), Enter to continue[/dim]")
        while True:
            cmd = console.input("[yellow]inspect> [/yellow]")
            if not cmd.strip():
                break
            try:
                token_idx, layer_idx = map(int, cmd.split(","))
                if 0 <= token_idx < len(toks) and 0 <= layer_idx < phi.num_layers * 2:
                    console.print(show_detail(toks, feat, token_idx, layer_idx, labels))
                else:
                    console.print("[red]Out of range[/red]")
            except:
                console.print("[red]Format: token_idx,layer_idx[/red]")

    console.print("[bold]Bye![/bold]")


if __name__ == "__main__":
    main()
