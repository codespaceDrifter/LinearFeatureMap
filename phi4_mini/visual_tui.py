"""
Visual TUI - shows top firing feature LABEL at each (layer, token) cell.
Scrollable grid to trace concepts through layers.
"""
import json
import os
import sys
import curses

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phi4_mini.feature_inference import Phi4FeatureInference
from scripts.config import config, pathconfig

FEATURE_THRESHOLD = 0.5
CELL_WIDTH = 20  # chars per cell for label
LAYER_COL_WIDTH = 8  # "L00 mlp" column


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

    return labels


def get_top_feature_label(features, hook_idx, token_idx, labels, num_layers):
    """Get the label of the top firing feature for a cell"""
    if features is None:
        return ""

    z = features[hook_idx, token_idx]  # (12288,)
    max_val = z.max().item()

    if max_val < FEATURE_THRESHOLD:
        return ""

    max_idx = z.argmax().item()

    # figure out layer type and number
    # hook_idx = layer * 2 for att, layer * 2 + 1 for mlp
    layer_type = "att" if hook_idx % 2 == 0 else "mlp"
    layer_num = hook_idx // 2

    # att only has SAEs for layers 1-31, mlp only for layers 0-30
    if layer_type == "att" and layer_num == 0:
        return ""  # no SAE at att[0]
    if layer_type == "mlp" and layer_num == num_layers - 1:
        return ""  # no SAE at mlp[31]

    # get label
    layer_labels = labels.get(layer_type, {}).get(layer_num, {})
    label = layer_labels.get(str(max_idx), {}).get("interpretation", f"f{max_idx}")

    return label[:CELL_WIDTH-1]  # truncate


def build_grid(tokens, features, labels, num_layers):
    """Build grid of labels: grid[row][col] = label string"""
    T = len(tokens)
    num_rows = num_layers * 2  # mlp + att per layer

    grid = []
    row_labels = []

    # rows from top (layer 31) to bottom (layer 0)
    for layer_num in range(num_layers - 1, -1, -1):
        # mlp row
        mlp_idx = layer_num * 2 + 1
        mlp_row = []
        for t in range(T):
            label = get_top_feature_label(features, mlp_idx, t, labels, num_layers)
            mlp_row.append(label)
        grid.append(mlp_row)
        row_labels.append(f"L{layer_num:02d}mlp")

        # att row
        att_idx = layer_num * 2
        att_row = []
        for t in range(T):
            label = get_top_feature_label(features, att_idx, t, labels, num_layers)
            att_row.append(label)
        grid.append(att_row)
        row_labels.append(f"L{layer_num:02d}att")

    return grid, row_labels, tokens


def run_viewer(stdscr, tokens, grid, row_labels):
    """Curses viewer with scroll"""
    curses.curs_set(0)
    curses.start_color()
    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_CYAN, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)

    scroll_y = 0  # row offset
    scroll_x = 0  # col offset

    num_rows = len(grid)
    num_cols = len(tokens)

    while True:
        stdscr.clear()
        height, width = stdscr.getmaxyx()

        # how many cells fit
        visible_cols = (width - LAYER_COL_WIDTH) // CELL_WIDTH
        visible_rows = height - 3  # header + footer

        # draw header row (tokens)
        header = " " * LAYER_COL_WIDTH
        for c in range(scroll_x, min(scroll_x + visible_cols, num_cols)):
            tok = tokens[c][:CELL_WIDTH-1].center(CELL_WIDTH)
            header += tok
        stdscr.addstr(0, 0, header[:width-1], curses.color_pair(2) | curses.A_BOLD)

        # draw token indices
        idx_row = " " * LAYER_COL_WIDTH
        for c in range(scroll_x, min(scroll_x + visible_cols, num_cols)):
            idx_row += f"[{c}]".center(CELL_WIDTH)
        stdscr.addstr(1, 0, idx_row[:width-1], curses.color_pair(3))

        # draw grid
        for r in range(visible_rows):
            row_idx = scroll_y + r
            if row_idx >= num_rows:
                break

            # row label
            line = row_labels[row_idx].ljust(LAYER_COL_WIDTH)

            # cells
            for c in range(scroll_x, min(scroll_x + visible_cols, num_cols)):
                cell = grid[row_idx][c]
                if cell:
                    cell_str = cell[:CELL_WIDTH-1].ljust(CELL_WIDTH)
                else:
                    cell_str = ".".center(CELL_WIDTH)
                line += cell_str

            y = r + 2
            if y < height - 1:
                # color cells with content
                stdscr.addstr(y, 0, row_labels[row_idx].ljust(LAYER_COL_WIDTH), curses.A_BOLD)
                x = LAYER_COL_WIDTH
                for c in range(scroll_x, min(scroll_x + visible_cols, num_cols)):
                    cell = grid[row_idx][c]
                    if x + CELL_WIDTH > width:
                        break
                    if cell:
                        stdscr.addstr(y, x, cell[:CELL_WIDTH-1].ljust(CELL_WIDTH), curses.color_pair(1))
                    else:
                        stdscr.addstr(y, x, ".".center(CELL_WIDTH))
                    x += CELL_WIDTH

        # footer
        footer = f"[arrows/hjkl: scroll | q: quit] rows {scroll_y+1}-{min(scroll_y+visible_rows, num_rows)}/{num_rows} cols {scroll_x+1}-{min(scroll_x+visible_cols, num_cols)}/{num_cols}"
        stdscr.addstr(height-1, 0, footer[:width-1], curses.A_DIM)

        stdscr.refresh()

        # input
        key = stdscr.getch()
        if key == ord('q'):
            break
        elif key == curses.KEY_UP or key == ord('k'):
            scroll_y = max(0, scroll_y - 1)
        elif key == curses.KEY_DOWN or key == ord('j'):
            scroll_y = min(num_rows - visible_rows, scroll_y + 1)
        elif key == curses.KEY_LEFT or key == ord('h'):
            scroll_x = max(0, scroll_x - 1)
        elif key == curses.KEY_RIGHT or key == ord('l'):
            scroll_x = min(num_cols - visible_cols, scroll_x + 1)
        elif key == curses.KEY_PPAGE:  # page up
            scroll_y = max(0, scroll_y - visible_rows)
        elif key == curses.KEY_NPAGE:  # page down
            scroll_y = min(num_rows - visible_rows, scroll_y + visible_rows)


def main():
    print("Loading Phi4 + SAEs...")
    phi = Phi4FeatureInference()
    labels = load_labels()
    print(f"Loaded labels for {len(labels['mlp'])} mlp + {len(labels['att'])} att layers")
    print("Ready!\n")

    while True:
        prompt = input(">>> ").strip()
        if prompt.lower() in ["quit", "exit", "q"]:
            break
        if not prompt:
            continue

        print("Generating...")
        responses, tokens, features = phi.generate([prompt], max_new_tokens=64)

        toks = tokens[0]
        feat = features[0]

        print(f"Response: {responses[0]}\n")
        print(f"Tokens: {len(toks)}, launching viewer...")

        # build grid
        grid, row_labels, _ = build_grid(toks, feat, labels, phi.num_layers)

        # run curses viewer
        curses.wrapper(lambda stdscr: run_viewer(stdscr, toks, grid, row_labels))

        print("\nBack to prompt. Enter another or 'q' to quit.\n")

    print("Bye!")


if __name__ == "__main__":
    main()
