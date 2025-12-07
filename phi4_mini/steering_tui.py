import json
import os
import sys

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich import box

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phi4_mini.steering_inference import Phi4SteeringInference
from scripts.config import config, pathconfig

console = Console()
BLOCK_WIDTH = 12
FEATURE_THRESHOLD = 0.75


def load_labels():
    """Load interpretation labels for all SAEs"""
    labels = {"mlp": {}, "att": {}}
    num_layers = config["num_layers"]

    for layer in range(num_layers - 1):
        path = pathconfig["interpretations"]["mlp"][layer]
        if os.path.exists(path):
            with open(path, "r") as f:
                labels["mlp"][layer] = json.load(f)

    for layer in range(1, num_layers):
        path = pathconfig["interpretations"]["att"][layer]
        if os.path.exists(path):
            with open(path, "r") as f:
                labels["att"][layer] = json.load(f)

    console.print(f"[dim]Loaded {len(labels['mlp'])} mlp + {len(labels['att'])} att label files[/dim]")
    return labels


def get_firing_features(features, hook_idx, token_idx, threshold=0.0):
    """Get features above threshold for a specific cell."""
    if features is None:
        return []

    z = features[hook_idx, token_idx]
    mask = z > threshold
    if not mask.any():
        return []

    indices = mask.nonzero().squeeze(-1).tolist()
    if isinstance(indices, int):
        indices = [indices]

    return [(idx, z[idx].item()) for idx in indices]


def format_token(tok):
    return tok[:BLOCK_WIDTH-1].center(BLOCK_WIDTH-1)


def format_cell(num_features, has_override=False):
    """Format cell: feature count, highlight if has override"""
    if has_override:
        return f"[bold red]{num_features}f*[/bold red]"
    if num_features > 0:
        return f"[green]{num_features}f[/green]"
    return "[dim]0[/dim]"


def build_tables(tokens, features, num_layers, blocks_per_row, overrides=None):
    """Build rich tables, one per wrap segment"""
    tables = []
    T = len(tokens)
    num_wraps = (T + blocks_per_row - 1) // blocks_per_row

    feature_counts = {}
    override_positions = set()

    if features is not None:
        fired = (features > FEATURE_THRESHOLD).sum(dim=-1)
        for hook_idx in range(num_layers * 2):
            for t in range(T):
                feature_counts[(hook_idx, t)] = fired[hook_idx, t].item()

    if overrides:
        for (layer, kind, token_idx) in overrides.keys():
            hook_idx = layer * 2 if kind == "att" else layer * 2 + 1
            override_positions.add((hook_idx, token_idx))

    for wrap_idx in range(num_wraps):
        tok_start = wrap_idx * blocks_per_row
        tok_end = min(tok_start + blocks_per_row, T)
        wrap_tokens = tokens[tok_start:tok_end]

        table = Table(box=box.SIMPLE, padding=0, collapse_padding=True)

        table.add_column("", style="bold", width=10)
        for i, t in enumerate(wrap_tokens):
            table.add_column(f"[{tok_start + i}] {format_token(t)}", width=BLOCK_WIDTH)

        for layer_num in range(num_layers - 1, -1, -1):
            mlp_idx = layer_num * 2 + 1
            row = [f"L{layer_num:02d} mlp"]
            for t in range(tok_start, tok_end):
                count = feature_counts.get((mlp_idx, t), 0)
                has_override = (mlp_idx, t) in override_positions
                row.append(format_cell(count, has_override))
            table.add_row(*row)

            att_idx = layer_num * 2
            row = [f"L{layer_num:02d} att"]
            for t in range(tok_start, tok_end):
                count = feature_counts.get((att_idx, t), 0)
                has_override = (att_idx, t) in override_positions
                row.append(format_cell(count, has_override))
            table.add_row(*row)

        tables.append(table)

    return tables


def show_features(tokens, features, token_idx, layer_idx, labels, phi, show_top=30):
    """Show features at a cell with editing UI"""
    layer_type = "att" if layer_idx % 2 == 0 else "mlp"
    layer_num = layer_idx // 2

    console.print(f"\n[bold]Layer {layer_num} {layer_type}_in | Token {token_idx}: '{tokens[token_idx]}'[/bold]\n")

    layer_labels = labels.get(layer_type, {}).get(layer_num, {})

    fires = get_firing_features(features, layer_idx, token_idx, threshold=0.0)
    fires.sort(key=lambda x: -x[1])

    # check for existing override
    override = phi.get_override(layer_num, layer_type, token_idx)
    if override is not None:
        console.print("[bold red]Override active for this position[/bold red]\n")

    # show top features
    console.print(f"[bold]Top {show_top} features:[/bold]")
    for i, (fid, fval) in enumerate(fires[:show_top]):
        label = layer_labels.get(str(fid), {}).get("interpretation", "[no label]")
        override_marker = ""
        if override is not None and override[fid].item() != 0:
            override_marker = f" [red](override: {override[fid].item():.2f})[/red]"
        console.print(f"  {i:2d}. [yellow]f{fid}[/yellow] = {fval:.3f}{override_marker}: {label}")

    return fires, layer_num, layer_type


def steering_menu(phi, tokens, features, labels):
    """Interactive menu for setting feature overrides"""
    while True:
        console.print("\n[bold cyan]Steering Menu[/bold cyan]")
        console.print("  [dim]1.[/dim] Select cell to edit (token,layer)")
        console.print("  [dim]2.[/dim] List current overrides")
        console.print("  [dim]3.[/dim] Clear all overrides")
        console.print("  [dim]4.[/dim] Re-run with steering")
        console.print("  [dim]q.[/dim] Back to main")

        cmd = Prompt.ask("Choice", default="1")

        if cmd == "q":
            return None

        elif cmd == "1":
            try:
                cell = Prompt.ask("Enter token_idx,layer_idx")
                token_idx, layer_idx = map(int, cell.split(","))

                if not (0 <= token_idx < len(tokens) and 0 <= layer_idx < phi.num_layers * 2):
                    console.print("[red]Out of range[/red]")
                    continue

                fires, layer_num, layer_type = show_features(tokens, features, token_idx, layer_idx, labels, phi)

                # edit loop
                while True:
                    console.print("\n[dim]Enter: 'f1234=5.0' to set, 'f1234' to see, 'clear' to clear this cell, 'done' to finish[/dim]")
                    edit_cmd = Prompt.ask("Edit")

                    if edit_cmd == "done":
                        break
                    elif edit_cmd == "clear":
                        key = (layer_num, layer_type, token_idx)
                        if key in phi.overrides:
                            del phi.overrides[key]
                            console.print("[green]Override cleared[/green]")
                        else:
                            console.print("[dim]No override to clear[/dim]")
                    elif "=" in edit_cmd:
                        try:
                            feat_part, val_part = edit_cmd.split("=")
                            feat_idx = int(feat_part.strip().lstrip("f"))
                            value = float(val_part.strip())
                            phi.set_override(layer_num, layer_type, token_idx, feature_idx=feat_idx, value=value)
                            console.print(f"[green]Set f{feat_idx} = {value}[/green]")
                        except:
                            console.print("[red]Invalid format. Use: f1234=5.0[/red]")
                    elif edit_cmd.startswith("f"):
                        try:
                            feat_idx = int(edit_cmd.lstrip("f"))
                            feat_val = features[layer_idx, token_idx, feat_idx].item()
                            label = labels.get(layer_type, {}).get(layer_num, {}).get(str(feat_idx), {}).get("interpretation", "[no label]")
                            override = phi.get_override(layer_num, layer_type, token_idx)
                            override_val = override[feat_idx].item() if override is not None else None
                            console.print(f"[yellow]f{feat_idx}[/yellow] = {feat_val:.3f}")
                            if override_val is not None and override_val != 0:
                                console.print(f"  [red]Override: {override_val:.2f}[/red]")
                            console.print(f"  {label}")
                        except:
                            console.print("[red]Invalid feature index[/red]")
                    elif edit_cmd.strip():
                        # search labels
                        query = edit_cmd.lower()
                        matches = []
                        for fid_str, data in labels.get(layer_type, {}).get(layer_num, {}).items():
                            interp = data.get("interpretation", "").lower()
                            if query in interp:
                                fid = int(fid_str)
                                fval = features[layer_idx, token_idx, fid].item()
                                matches.append((fid, fval, data.get("interpretation", "")))

                        if matches:
                            matches.sort(key=lambda x: -x[1])
                            console.print(f"\n[bold]Features matching '{edit_cmd}':[/bold]")
                            for fid, fval, interp in matches[:10]:
                                console.print(f"  [yellow]f{fid}[/yellow] = {fval:.3f}: {interp}")
                        else:
                            console.print(f"[dim]No labels matching '{edit_cmd}'[/dim]")

            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")

        elif cmd == "2":
            if not phi.overrides:
                console.print("[dim]No overrides set[/dim]")
            else:
                console.print("\n[bold]Current overrides:[/bold]")
                for (layer, kind, token_idx), vec in phi.overrides.items():
                    nonzero = (vec != 0).sum().item()
                    top_feats = vec.abs().topk(min(3, int(nonzero)))
                    top_str = ", ".join([f"f{i}={vec[i].item():.1f}" for i in top_feats.indices.tolist() if vec[i] != 0])
                    console.print(f"  L{layer} {kind} t{token_idx}: {nonzero} features modified ({top_str})")

        elif cmd == "3":
            phi.clear_overrides()
            console.print("[green]All overrides cleared[/green]")

        elif cmd == "4":
            return "rerun"


def main():
    console.print("[bold]Loading Phi4 + SAEs (steering mode)...[/bold]")
    try:
        phi = Phi4SteeringInference()
    except AssertionError as e:
        console.print(f"[red]Error loading SAEs:[/red]\n{e}")
        return

    labels = load_labels()
    console.print("[green]Ready![/green]\n")

    current_prompt = None
    current_response = None
    current_tokens = None
    current_features = None

    while True:
        if current_prompt is None:
            prompt = console.input("[bold cyan]>>> [/bold cyan]")

            if prompt.strip().lower() in ["quit", "exit", "q"]:
                break
            if not prompt.strip():
                continue

            current_prompt = prompt
        else:
            prompt = current_prompt

        console.print("[dim]Generating...[/dim]")

        if phi.overrides:
            responses, tokens, features = phi.generate_steered([prompt], max_new_tokens=64)
        else:
            responses, tokens, features = phi.generate([prompt], max_new_tokens=64)

        current_response = responses[0]
        current_tokens = tokens[0]
        current_features = features[0]

        console.clear()
        console.print(f"[bold]Prompt:[/bold] {prompt}")
        console.print(f"[bold]Response:[/bold] {current_response}")
        if phi.overrides:
            console.print(f"[bold red]Steering active: {len(phi.overrides)} position(s) modified[/bold red]")
        console.print()

        blocks_per_row = max(1, (console.width - 12) // BLOCK_WIDTH)
        tables = build_tables(current_tokens, current_features, phi.num_layers, blocks_per_row, phi.overrides)
        for table in tables:
            console.print(table)
            console.print()

        console.print("[dim]Cells: [green]Nf[/green]=features firing, [red]*[/red]=has override[/dim]")
        console.print("[dim]Commands: 'token,layer' to inspect, 's' for steering menu, 'n' for new prompt[/dim]")

        while True:
            cmd = console.input("[yellow]> [/yellow]")

            if cmd.strip().lower() == "n":
                current_prompt = None
                phi.clear_overrides()
                break
            elif cmd.strip().lower() == "s":
                result = steering_menu(phi, current_tokens, current_features, labels)
                if result == "rerun":
                    break  # re-run with same prompt
                else:
                    # redraw
                    console.clear()
                    console.print(f"[bold]Prompt:[/bold] {prompt}")
                    console.print(f"[bold]Response:[/bold] {current_response}")
                    if phi.overrides:
                        console.print(f"[bold red]Steering active: {len(phi.overrides)} position(s) modified[/bold red]")
                    console.print()
                    for table in tables:
                        console.print(table)
                        console.print()
                    console.print("[dim]Cells: [green]Nf[/green]=features firing, [red]*[/red]=has override[/dim]")
                    console.print("[dim]Commands: 'token,layer' to inspect, 's' for steering menu, 'n' for new prompt[/dim]")
            elif "," in cmd:
                try:
                    token_idx, layer_idx = map(int, cmd.split(","))
                    if 0 <= token_idx < len(current_tokens) and 0 <= layer_idx < phi.num_layers * 2:
                        show_features(current_tokens, current_features, token_idx, layer_idx, labels, phi)
                    else:
                        console.print("[red]Out of range[/red]")
                except:
                    console.print("[red]Format: token_idx,layer_idx[/red]")
            elif not cmd.strip():
                continue
            else:
                console.print("[dim]Unknown command[/dim]")

    console.print("[bold]Bye![/bold]")


if __name__ == "__main__":
    main()
