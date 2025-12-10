"""
Feature visualization TUI for Phi4-mini with SAE features.

Shows labeled features flowing through the model:
- Grid: rows = SAE positions (att/mlp per layer), columns = tokens
- Only shows cells with features firing > 0.75 threshold
- Displays TOP labeled feature in each cell
- Sidebar shows full feature info on hover

pip install textual
"""
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from phi4_mini.feature_inference import Phi4FeatureInference
from scripts.config import config, pathconfig

from textual.app import App, ComposeResult
from textual.widgets import Static, Header, Footer, Input, DataTable, Label
from textual.containers import Container, Horizontal, Vertical
from textual.binding import Binding
from textual.screen import Screen

FEATURE_THRESHOLD = 0.75


def load_labels():
    """Load interpretation labels for all SAEs."""
    labels = {"mlp": {}, "att": {}}
    num_layers = config["num_layers"]

    for layer in range(num_layers - 1):  # mlp 0-30
        path = pathconfig["interpretations"]["mlp"][layer]
        if os.path.exists(path):
            with open(path, "r") as f:
                labels["mlp"][layer] = json.load(f)

    for layer in range(1, num_layers):  # att 1-31
        path = pathconfig["interpretations"]["att"][layer]
        if os.path.exists(path):
            with open(path, "r") as f:
                labels["att"][layer] = json.load(f)

    return labels


class FeatureVizScreen(Screen):
    """Screen showing the feature grid."""

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("escape", "back", "Back"),
        Binding("n", "new_prompt", "New Prompt", show=True),
        Binding("p", "new_prompt", "New Prompt", show=False),
        Binding("h", "cursor_left", "Left", show=False),
        Binding("j", "cursor_down", "Down", show=False),
        Binding("k", "cursor_up", "Up", show=False),
        Binding("l", "cursor_right", "Right", show=False),
    ]

    def action_cursor_left(self) -> None:
        self.query_one("#grid", DataTable).action_cursor_left()

    def action_cursor_right(self) -> None:
        self.query_one("#grid", DataTable).action_cursor_right()

    def action_cursor_up(self) -> None:
        self.query_one("#grid", DataTable).action_cursor_up()

    def action_cursor_down(self) -> None:
        self.query_one("#grid", DataTable).action_cursor_down()

    def __init__(self, phi, tokens, features, response, labels):
        super().__init__()
        self.phi = phi
        self.tokens = tokens
        self.features = features  # (64, seq_len, 12288)
        self.response = response
        self.labels = labels
        self.num_layers = phi.num_layers

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label(
            f"[bold]Response:[/bold] [magenta]{self.response[:100]}{'...' if len(self.response) > 100 else ''}[/]  |  [dim]{len(self.tokens)} tokens, n/p: new prompt[/dim]",
            id="info-label"
        )
        yield Horizontal(
            DataTable(id="grid"),
            Vertical(
                Static("[bold]Feature Info[/bold]", id="sidebar-title"),
                Static("[dim]Select a cell to view details[/dim]", id="sidebar-content"),
                id="sidebar",
            ),
            id="main-container",
        )
        yield Footer()

    def _get_top_feature(self, hook_idx, token_idx):
        """Get top firing feature above threshold with its label."""
        z = self.features[hook_idx, token_idx]  # (12288,)
        max_val = z.max().item()

        if max_val < FEATURE_THRESHOLD:
            return None

        max_idx = z.argmax().item()

        # Determine layer type and number
        # hook_idx = layer * 2 for att, layer * 2 + 1 for mlp
        layer_type = "att" if hook_idx % 2 == 0 else "mlp"
        layer_num = hook_idx // 2

        # Get label
        layer_labels = self.labels.get(layer_type, {}).get(layer_num, {})
        label = layer_labels.get(str(max_idx), {}).get("interpretation", None)

        return {
            "idx": max_idx,
            "val": max_val,
            "label": label,
            "layer_type": layer_type,
            "layer_num": layer_num,
        }

    def _get_all_firing_features(self, hook_idx, token_idx):
        """Get all features firing above threshold."""
        z = self.features[hook_idx, token_idx]  # (12288,)
        mask = z > FEATURE_THRESHOLD

        if not mask.any():
            return []

        indices = mask.nonzero().squeeze(-1).tolist()
        if isinstance(indices, int):
            indices = [indices]

        layer_type = "att" if hook_idx % 2 == 0 else "mlp"
        layer_num = hook_idx // 2
        layer_labels = self.labels.get(layer_type, {}).get(layer_num, {})

        results = []
        for idx in indices:
            val = z[idx].item()
            label = layer_labels.get(str(idx), {}).get("interpretation", None)
            results.append({"idx": idx, "val": val, "label": label})

        results.sort(key=lambda x: -x["val"])
        return results

    def on_mount(self) -> None:
        table = self.query_one("#grid", DataTable)
        table.cursor_type = "cell"
        table.zebra_stripes = True

        # Columns: SAE position + each token
        table.add_column("SAE", key="sae")
        for i, tok in enumerate(self.tokens):
            display_tok = tok.replace("\n", "\\n").replace("\t", "\\t")[:6]
            table.add_column(f"[{i}]{display_tok}", key=f"t{i}")

        # Rows: from top (layer 31) to bottom (layer 0)
        # Order: L31 mlp, L31 att, L30 mlp, L30 att, ...
        for layer_num in range(self.num_layers - 1, -1, -1):
            # MLP row (hook_idx = layer * 2 + 1)
            mlp_idx = layer_num * 2 + 1
            row = [f"L{layer_num:02d} mlp"]
            for t_idx in range(len(self.tokens)):
                top = self._get_top_feature(mlp_idx, t_idx)
                if top and top["label"]:
                    # Truncate label to fit cell
                    short_label = top["label"][:12]
                    row.append(f"[green]{short_label}[/]")
                elif top:
                    row.append(f"[yellow]f{top['idx']}[/]")
                else:
                    row.append("[dim]·[/]")
            table.add_row(*row, key=f"L{layer_num:02d}_mlp")

            # ATT row (hook_idx = layer * 2)
            att_idx = layer_num * 2
            row = [f"L{layer_num:02d} att"]
            for t_idx in range(len(self.tokens)):
                top = self._get_top_feature(att_idx, t_idx)
                if top and top["label"]:
                    short_label = top["label"][:12]
                    row.append(f"[cyan]{short_label}[/]")
                elif top:
                    row.append(f"[yellow]f{top['idx']}[/]")
                else:
                    row.append("[dim]·[/]")
            table.add_row(*row, key=f"L{layer_num:02d}_att")

    def on_data_table_cell_highlighted(self, event: DataTable.CellHighlighted) -> None:
        if event.coordinate.column == 0:
            self._update_sidebar(None, None)
            return

        row_key = event.cell_key.row_key.value
        col_idx = event.coordinate.column - 1

        if col_idx < 0 or col_idx >= len(self.tokens):
            return

        # Parse row key to get hook_idx
        # Format: "L{layer:02d}_mlp" or "L{layer:02d}_att"
        layer_num = int(row_key[1:3])
        layer_type = row_key[4:]  # "mlp" or "att"
        hook_idx = layer_num * 2 + (1 if layer_type == "mlp" else 0)

        self._update_sidebar(hook_idx, col_idx)

    def _update_sidebar(self, hook_idx, col_idx):
        """Update sidebar with feature info."""
        sidebar = self.query_one("#sidebar-content", Static)

        if hook_idx is None:
            sidebar.update("[dim]Select a cell to view details[/dim]")
            return

        layer_type = "att" if hook_idx % 2 == 0 else "mlp"
        layer_num = hook_idx // 2

        tok = self.tokens[col_idx]
        tok_display = tok.replace("\n", "\\n").replace("\t", "\\t")

        lines = [
            f"[bold cyan]L{layer_num:02d} {layer_type}[/bold cyan]",
            "",
            f"[bold]Token {col_idx}:[/bold] '{tok_display}'",
            "",
        ]

        firing = self._get_all_firing_features(hook_idx, col_idx)

        if firing:
            lines.append(f"[bold green]Firing features (>{FEATURE_THRESHOLD}):[/bold green]")
            lines.append("")
            for i, feat in enumerate(firing[:10]):  # Top 10
                val_str = f"{feat['val']:.2f}"
                if feat["label"]:
                    lines.append(f"[yellow]f{feat['idx']}[/yellow] ({val_str})")
                    lines.append(f"  [green]{feat['label']}[/green]")
                else:
                    lines.append(f"[yellow]f{feat['idx']}[/yellow] ({val_str}) [dim]no label[/dim]")
                lines.append("")

            if len(firing) > 10:
                lines.append(f"[dim]... and {len(firing) - 10} more[/dim]")
        else:
            lines.append("[dim]No features firing above threshold[/dim]")

        sidebar.update("\n".join(lines))

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_new_prompt(self) -> None:
        self.app.pop_screen()

    def action_quit(self) -> None:
        self.app.exit()


class PromptScreen(Screen):
    """Screen for entering prompts."""

    BINDINGS = [
        Binding("escape", "quit", "Quit"),
    ]

    def __init__(self, phi, labels):
        super().__init__()
        self.phi = phi
        self.labels = labels

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Static(
                "[bold]Phi4-mini Feature Visualizer[/bold]\n\n"
                "See SAE features firing through the model.\n"
                "Only shows features > 0.75 threshold with labels.\n",
                id="intro"
            ),
            Input(placeholder="Enter prompt...", id="prompt-input"),
            Static("", id="status"),
            id="prompt-container",
        )
        yield Footer()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        prompt = event.value.strip()
        if not prompt:
            return

        status = self.query_one("#status", Static)
        status.update("[dim]Generating...[/dim]")

        responses, tokens, features = self.phi.generate([prompt], max_new_tokens=64)

        response = responses[0]
        toks = tokens[0]
        feat = features[0]

        status.update(f"[green]Done![/green] {len(toks)} tokens")
        self.query_one("#prompt-input", Input).value = ""

        self.app.push_screen(FeatureVizScreen(self.phi, toks, feat, response, self.labels))

    def action_quit(self) -> None:
        self.app.exit()


class FeatureVizApp(App):
    """Main app."""

    CSS = """
    #prompt-container {
        align: center middle;
        width: 100%;
        height: 100%;
    }

    #intro {
        text-align: center;
        margin-bottom: 2;
    }

    #prompt-input {
        width: 80%;
        margin: 1;
    }

    #status {
        text-align: center;
        margin-top: 1;
    }

    #info-label {
        dock: top;
        background: $surface;
        padding: 1;
        height: auto;
    }

    #main-container {
        height: 1fr;
        width: 100%;
    }

    #grid {
        width: 70%;
        height: 100%;
    }

    #sidebar {
        width: 30%;
        height: 100%;
        border-left: solid $primary;
        padding: 1;
        overflow-y: auto;
    }

    #sidebar-title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }

    #sidebar-content {
        height: auto;
    }
    """

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit"),
    ]

    def __init__(self, phi, labels):
        super().__init__()
        self.phi = phi
        self.labels = labels

    def on_mount(self) -> None:
        self.push_screen(PromptScreen(self.phi, self.labels))


def main():
    print("Loading Phi4-mini + SAEs...")
    try:
        phi = Phi4FeatureInference()
    except AssertionError as e:
        print(f"Error loading SAEs:\n{e}")
        return

    print("Loading feature labels...")
    labels = load_labels()
    mlp_count = len(labels.get("mlp", {}))
    att_count = len(labels.get("att", {}))
    print(f"Loaded {mlp_count} mlp + {att_count} att label files")

    print("Ready!")
    app = FeatureVizApp(phi, labels)
    app.run()


if __name__ == "__main__":
    main()
