"""Shared utilities for Lecture 08: counterfactual dataset design.

This module collects generic visualization, dataset-diagnostics, and
interchange-intervention helpers that are reused across the lecture notebook.
"""


import random
import sys
from pathlib import Path
from typing import Callable
import numpy as np

_repo_root = None
for p in [Path.cwd(), *Path.cwd().parents]:
    if (p / "pyproject.toml").exists() and (p / "causalab").is_dir():
        _repo_root = p
        break
    if (p / "causalab" / "pyproject.toml").exists() and (p / "causalab" / "causalab").is_dir():
        _repo_root = p / "causalab"
        break

if _repo_root is not None and str(_repo_root) not in sys.path:
    sys.path.append(str(_repo_root))

from causalab.causal.causal_model import CausalModel
from causalab.causal.causal_utils import can_distinguish_with_dataset
from causalab.causal.counterfactual_dataset import CounterfactualExample
from causalab.causal.trace import CausalTrace, Mechanism, input_var

import causalab.causal.causal_utils as _cu
if not hasattr(_cu, "statement_conjunction_function"):
    from typing import Sequence
    def statement_conjunction_function(statements: Sequence[str], delimiters: Sequence[str]) -> str:
        stmts = list(statements)
        if not stmts:
            return ""
        first = stmts[0]
        if first:
            first = first[0].upper() + first[1:]
        if len(delimiters) < 2:
            end = "."; sep = " "; last_sep = " "
        elif len(delimiters) == 2:
            sep = delimiters[0]; last_sep = delimiters[0]; end = delimiters[1]
        else:
            sep = delimiters[0]; last_sep = delimiters[-2]; end = delimiters[-1]
        if len(stmts) == 1:
            return f"{first}{end}"
        if len(stmts) == 2:
            return f"{first}{last_sep}{stmts[1]}{end}"
        middle = "".join(f"{sep}{s}" for s in stmts[1:-1])
        return f"{first}{middle}{last_sep}{stmts[-1]}{end}"
    _cu.statement_conjunction_function = statement_conjunction_function


try:
    import matplotlib.pyplot as plt
    import networkx as nx
    from matplotlib.patches import Patch

    GRAPH_DEFAULT_COLOR = "#D9EAD3"
    GRAPH_TARGET_COLOR = "#4C78A8"
    GRAPH_CONTROL_COLOR = "#B279A2"
    GRAPH_OUTPUT_COLOR = "#F58518"

    def _wrap_graph_label(label: str, width: int = 12) -> str:
        return textwrap.fill(str(label).replace("_", " "), width=width, break_long_words=False)

    def print_structure(
        model: CausalModel,
        *,
        highlight: dict[str, str] | None = None,
        font: int = 9,
        node_size: int = 1200,
        title: str | None = None,
        legend_items: list[tuple[str, str]] | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> None:
        graph = nx.DiGraph()
        graph.add_nodes_from(model.variables)
        graph.add_edges_from([(parent, child) for child in model.variables for parent in model.parents[child]])

        nodelist = list(model.variables)
        colors = [(highlight or {}).get(v, GRAPH_DEFAULT_COLOR) for v in nodelist]
        labels = {v: _wrap_graph_label(v) for v in nodelist}

        if figsize is None:
            width = max(6.2, 0.70 * len(nodelist) + 1.4)
            height = 3.4 if len(nodelist) <= 8 else 4.2
            figsize = (width, height)

        pos = getattr(model, "print_pos", None)
        if not pos:
            pos = nx.spring_layout(graph, seed=0)

        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

        nx.draw_networkx_nodes(
            graph,
            pos=pos,
            nodelist=nodelist,
            node_color=colors,
            node_size=node_size,
            edgecolors="#333333",
            linewidths=1.2,
            ax=ax,
        )
        nx.draw_networkx_edges(
            graph,
            pos=pos,
            nodelist=nodelist,
            arrows=True,
            arrowstyle="-|>",
            arrowsize=12,
            width=1.0,
            edge_color="#222222",
            ax=ax,
        )
        nx.draw_networkx_labels(
            graph,
            pos=pos,
            labels=labels,
            font_size=font,
            font_weight="semibold",
            ax=ax,
        )

        if title:
            ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
        ax.axis("off")

        if legend_items:
            handles = [Patch(facecolor=color, edgecolor="black", label=label) for label, color in legend_items]
            ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8.5)

        plt.show()
except Exception:
    GRAPH_DEFAULT_COLOR = "#D9EAD3"
    GRAPH_TARGET_COLOR = "#4C78A8"
    GRAPH_CONTROL_COLOR = "#B279A2"
    GRAPH_OUTPUT_COLOR = "#F58518"

    def print_structure(model: CausalModel, *_args, highlight=None, title=None, legend_items=None, **_kwargs) -> None:
        print("Causal graph edges:")
        if title:
            print(title)
        if legend_items:
            print("Legend:", ", ".join(f"{label}={color}" for label, color in legend_items))
        if highlight:
            print("  highlighted:", ", ".join(f"{k}={v}" for k, v in highlight.items()))
        for child in model.variables:
            for parent in model.parents[child]:
                print(f"  {parent} -> {child}")


def show_localization_view(
    model: CausalModel,
    *,
    target_vars: list[str],
    control_vars: list[str],
    title: str,
    output_var: str = "raw_output",
) -> None:
    highlight = {var: GRAPH_TARGET_COLOR for var in target_vars}
    for var in control_vars:
        highlight[var] = GRAPH_CONTROL_COLOR
    if output_var in getattr(model, "variables", []):
        highlight[output_var] = GRAPH_OUTPUT_COLOR

    legend_items = [("target variable", GRAPH_TARGET_COLOR)]
    if control_vars:
        legend_items.append(("control variable", GRAPH_CONTROL_COLOR))
    if output_var in getattr(model, "variables", []):
        legend_items.append(("final output", GRAPH_OUTPUT_COLOR))

    print_structure(
        model,
        highlight=highlight,
        title=title,
        legend_items=legend_items,
    )

def set_seed(seed: int = 0) -> None:
    random.seed(seed); np.random.seed(seed)

def make_dataset(n: int, sampler: Callable[[], CounterfactualExample]) -> list[CounterfactualExample]:
    return [sampler() for _ in range(n)]

def report_distinguishability(model, dataset, vars1, vars2, label):
    res = can_distinguish_with_dataset(dataset, model, vars1, model if vars2 is not None else None, vars2)
    print(f"{label}: distinguish {vars1} vs {vars2} -> {res['count']}/{len(dataset)} ({res['proportion']:.3f})")
    return res

def proportion_true(dataset, var, *, which="input"):
    traces = [ex["input"] for ex in dataset] if which == "input" else [ex["counterfactual_inputs"][0] for ex in dataset]
    vals = [t[var] for t in traces]
    if not all(isinstance(v, (bool, np.bool_)) for v in vals):
        raise TypeError(f"{var} is not Boolean in traces")
    return float(sum(bool(v) for v in vals)) / len(vals)

# ─── Dataset diagnostics + visualization helpers ────────────────────────────

from collections import Counter
import pandas as pd
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    _PLOTTING_AVAILABLE = True
    _PLOTTING_IMPORT_ERROR = None
except Exception as _plot_err:
    plt = None
    sns = None
    _PLOTTING_AVAILABLE = False
    _PLOTTING_IMPORT_ERROR = _plot_err

if _PLOTTING_AVAILABLE:
    sns.set_theme(style="whitegrid", context="notebook", font_scale=0.85)
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 9,
        "legend.fontsize": 8.5,
    })

def _ensure_plotting(context: str) -> bool:
    if _PLOTTING_AVAILABLE:
        return True
    print(f"(skip) {context}: plotting libraries unavailable ({_PLOTTING_IMPORT_ERROR})")
    return False

def _pretty_inline_label(label: str) -> str:
    return str(label).replace("_", " ")

def _pretty_wrap_label(label: str, width: int = 16) -> str:
    text = str(label).replace("Δ", "Δ ").replace("_", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return textwrap.fill(text, width=width, break_long_words=False, break_on_hyphens=False)

def _pretty_metric_label(label: str, width: int = 18) -> str:
    text = str(label).replace("_", " ")
    m = re.fullmatch(r"patch\((.+)\)", text)
    if m:
        inner_raw = m.group(1)
        # If the whole string fits, keep it on one line.
        if len(text) <= width:
            return text
        inner_width = max(8, width - 6)
        inner = textwrap.fill(inner_raw, width=inner_width, break_long_words=False, break_on_hyphens=False)
        return f"patch(\n{inner}\n)"
    return _pretty_wrap_label(text, width=width)

def _annot_fontsize(mat) -> int:
    cells = int(np.prod(mat.shape))
    if cells <= 9:
        return 11
    if cells <= 18:
        return 10
    if cells <= 40:
        return 9
    if cells <= 80:
        return 8
    return 7

def _iia_annot_fontsize(mat) -> float:
    cells = int(np.prod(mat.shape))
    if cells <= 24:
        return 13
    if cells <= 48:
        return 12
    if cells <= 96:
        return 10.5
    if cells <= 160:
        return 9.5
    return 8.5

def _heatmap_figsize(mat: pd.DataFrame, *, min_width: float = 4.8, min_height: float = 2.4, max_width: float = 9.0, max_height: float = 4.2) -> tuple[float, float]:
    """Compact sizing for the small (≤ 4 rows × ≤ 6 cols) diagnostic heatmaps we use in this lecture.

    Heuristic: columns dominate width, rows dominate height; label lengths add a small
    constant. We clamp to [min, max] to avoid either tiny or oversized figures.
    """
    n_rows, n_cols = mat.shape
    row_len = max((len(str(idx).replace("\n", " ")) for idx in mat.index), default=0)
    col_len = max((len(str(col).replace("\n", " ")) for col in mat.columns), default=0)
    width = 1.4 + 0.95 * n_cols + 0.035 * row_len
    height = 1.1 + 0.42 * n_rows + 0.010 * col_len
    width = min(max_width, max(min_width, width))
    height = min(max_height, max(min_height, height))
    return width, height

def _style_heatmap_axes(ax, *, x_rotation=0):
    ax.tick_params(axis="x", rotation=x_rotation, labelsize=9, pad=4)
    ax.tick_params(axis="y", rotation=0, labelsize=9, pad=3)
    for label in ax.get_xticklabels():
        label.set_ha("center")
    for label in ax.get_yticklabels():
        label.set_va("center")


def _any_value_changed(base, cf, vars):
    return any(base[v] != cf[v] for v in vars)

def _tuple_value(t, vars):
    return t[vars[0]] if len(vars) == 1 else tuple(t[v] for v in vars)

def classify_pair_for_hypothesis(model, ex, vars1, vars2):
    base = ex["input"]; cf = ex["counterfactual_inputs"][0]
    y0 = base["raw_output"]
    y1 = model.run_interchange(base, {v: cf for v in vars1})["raw_output"]
    delta1 = _any_value_changed(base, cf, vars1)
    if vars2 is None:
        return {"category": "effect (changes output)" if y1 != y0 else "no effect (matches baseline)", "y_base": y0, "y_vars1": y1, "delta_vars1": delta1}
    y2 = model.run_interchange(base, {v: cf for v in vars2})["raw_output"]
    delta2 = _any_value_changed(base, cf, vars2)
    cat = "clean: distinguishable (vars1 != vars2)" if y1 != y2 else ("confounded: indistinguishable (both match baseline)" if y1 == y0 else "confounded: indistinguishable (both same non-baseline)")
    return {"category": cat, "y_base": y0, "y_vars1": y1, "y_vars2": y2, "delta_vars1": delta1, "delta_vars2": delta2}

def plot_boolean_rate(dataset, var, *, which="input", title=None):
    if not _ensure_plotting("plot_boolean_rate"):
        return
    vals = [bool((ex["input"] if which == "input" else ex["counterfactual_inputs"][0])[var]) for ex in dataset]
    df = pd.DataFrame({var: vals})
    counts = df[var].value_counts().rename_axis(var).reset_index(name="count")
    counts["percent"] = 100 * counts["count"] / len(df)
    counts[var] = counts[var].map({False: "False", True: "True"})
    counts[var] = pd.Categorical(counts[var], categories=["False", "True"], ordered=True)
    counts = counts.sort_values(var)
    fig, ax = plt.subplots(figsize=(4.5, 3))
    sns.barplot(data=counts, x=var, y="percent", ax=ax, color="#54A24B")
    ax.set_ylim(0, 100); ax.set_ylabel("% of examples"); ax.set_xlabel(""); ax.set_title(title or f"{which}: {var}")
    for i, row in enumerate(counts.itertuples(index=False)):
        ax.text(i, row.percent + 1, f"{row.percent:.1f}% ({int(row.count)})", ha="center", va="bottom")
    plt.tight_layout(); plt.show()

def plot_boolean_contingency(dataset, var_x, var_y, *, which="input", title=None, normalize=False):
    if not _ensure_plotting("plot_boolean_contingency"):
        return
    rows = [{"x": bool((ex["input"] if which == "input" else ex["counterfactual_inputs"][0])[var_x]),
             "y": bool((ex["input"] if which == "input" else ex["counterfactual_inputs"][0])[var_y])} for ex in dataset]
    df = pd.DataFrame(rows).rename(columns={"x": var_x, "y": var_y})
    mat = pd.crosstab(df[var_x], df[var_y])
    if normalize:
        mat = 100 * mat / mat.values.sum(); fmt = ".1f"; cbar_kws = {"label": "% of pairs"}; cmap = "viridis"
    else:
        fmt = "d"; cbar_kws = {"label": "count"}; cmap = "Blues"
    mat = mat.reindex(index=[False, True], columns=[False, True])
    fig, ax = plt.subplots(figsize=(4.8, 3.6))
    sns.heatmap(mat, annot=True, fmt=fmt, cmap=cmap, cbar=True, cbar_kws=cbar_kws, ax=ax)
    ax.set_title(title or f"{which}: {var_x} vs {var_y}"); ax.set_xlabel(var_y); ax.set_ylabel(var_x)
    plt.tight_layout(); plt.show()

def plot_category_bar(df, title):
    if not _ensure_plotting("plot_category_bar"):
        return
    counts = df["category"].value_counts().rename_axis("category").reset_index(name="count")
    counts["percent"] = 100 * counts["count"] / len(df)
    preferred = ["clean: distinguishable (vars1 != vars2)", "confounded: indistinguishable (both match baseline)",
                 "confounded: indistinguishable (both same non-baseline)", "effect (changes output)", "no effect (matches baseline)"]
    order = [c for c in preferred if c in set(counts["category"])] + [c for c in counts["category"].tolist() if c not in preferred]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    sns.barplot(data=counts, y="category", x="percent", order=order, ax=ax, color="#4C78A8")
    ax.set_xlim(0, 100); ax.set_xlabel("% of pairs"); ax.set_ylabel(""); ax.set_title(title)
    import textwrap
    ax.set_yticklabels([textwrap.fill(t.get_text(), width=44) for t in ax.get_yticklabels()])
    lookup = {r.category: (float(r.percent), int(r.count)) for r in counts.itertuples(index=False)}
    for i, cat in enumerate(order):
        p, n = lookup[cat]; ax.text(p + 1, i, f"{p:.1f}% ({n})", va="center")
    plt.tight_layout(); plt.show()

def plot_confound_breakdown(df, title, *, vars1_label="vars1", vars2_label="vars2"):
    if not _ensure_plotting("plot_confound_breakdown"):
        return
    if "delta_vars2" not in df.columns: return
    conf = df[df["category"].astype(str).str.startswith("confounded")].copy()
    if conf.empty: return
    def _detail(r):
        same_as_base = bool(r["y_vars1"] == r["y_base"])
        return f"{'baseline' if same_as_base else 'non-baseline'} | Δ{vars1_label}={int(bool(r['delta_vars1']))} Δ{vars2_label}={int(bool(r['delta_vars2']))}"
    conf["confound_detail"] = conf.apply(_detail, axis=1)
    counts = conf["confound_detail"].value_counts().rename_axis("confound_detail").reset_index(name="count")
    counts["percent_of_confounded"] = 100 * counts["count"] / len(conf)
    fig, ax = plt.subplots(figsize=(9, 3.2))
    sns.barplot(data=counts, y="confound_detail", x="percent_of_confounded", ax=ax, color="#F58518")
    ax.set_xlim(0, 100); ax.set_xlabel("% of confounded pairs"); ax.set_ylabel(""); ax.set_title(f"{title} (confound breakdown)")
    import textwrap
    ax.set_yticklabels([textwrap.fill(t.get_text(), width=44) for t in ax.get_yticklabels()])
    for i, row in enumerate(counts.itertuples(index=False)):
        ax.text(float(row.percent_of_confounded) + 1, i, f"{float(row.percent_of_confounded):.1f}% ({int(row.count)})", va="center")
    plt.tight_layout(); plt.show()

def plot_clean_rate_by_base_values(df, base_var1, base_var2, *, title):
    if not _ensure_plotting("plot_clean_rate_by_base_values"):
        return
    if not {"base_vars1", "base_vars2", "category"}.issubset(df.columns): return
    tmp = df.copy(); tmp["is_clean"] = tmp["category"].astype(str).str.startswith("clean")
    pivot = tmp.pivot_table(index="base_vars1", columns="base_vars2", values="is_clean", aggfunc="mean") * 100
    pivot = pivot.reindex(index=[False, True], columns=[False, True])
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    sns.heatmap(pivot, annot=True, fmt=".1f", vmin=0, vmax=100, cmap="viridis", ax=ax)
    ax.set_xlabel(f"base {base_var2}"); ax.set_ylabel(f"base {base_var1}"); ax.set_title(title)
    plt.tight_layout(); plt.show()

def summarize_hypothesis_dataset(model, dataset, vars1, vars2, name, *, show_plots=True, show_breakdown=True, max_examples_per_category=0):
    rows = []
    for ex in dataset:
        base = ex["input"]; cf = ex["counterfactual_inputs"][0]
        info = classify_pair_for_hypothesis(model, ex, vars1, vars2)
        row = dict(info); row["base_raw_input"] = base["raw_input"]; row["cf_raw_input"] = cf["raw_input"]
        row["base_vars1"] = _tuple_value(base, vars1); row["cf_vars1"] = _tuple_value(cf, vars1)
        if vars2 is not None:
            row["base_vars2"] = _tuple_value(base, vars2); row["cf_vars2"] = _tuple_value(cf, vars2)
        rows.append(row)
    df = pd.DataFrame(rows)
    counts = df["category"].value_counts(); total = len(df)
    print(f"\n{name}\n{'-' * len(name)}")
    for k, v in counts.items():
        print(f"  {k}: {v}/{total} ({v/total:.3f})")
    if show_plots: plot_category_bar(df, name)
    if vars2 is not None and show_breakdown:
        plot_confound_breakdown(df, name, vars1_label="+".join(vars1), vars2_label="+".join(vars2))
    if max_examples_per_category > 0:
        print("\nRepresentative examples:")
        for cat, sub in df.groupby("category"):
            print(f"\n[{cat}]")
            for _, r in sub.head(max_examples_per_category).iterrows():
                print("  base:", r["base_raw_input"]); print("  cf:  ", r["cf_raw_input"])
                line = f"  y_base={r['y_base']} | y_vars1={r['y_vars1']}"
                if "y_vars2" in r: line += f" | y_vars2={r['y_vars2']}"
                print(line)
    return df

def compute_testable_rate(model, dataset, vars1, vars2):
    total = len(dataset)
    if total == 0: return {"testable": 0, "total": 0, "percent": 0.0}
    res = can_distinguish_with_dataset(dataset, model, vars1, model if vars2 is not None else None, vars2)
    return {"testable": int(res["count"]), "total": total, "percent": 100 * float(res["count"]) / total}

def plot_testable_rate_by_dataset(model, datasets, vars1, vars2, *, title):
    if not _ensure_plotting("plot_testable_rate_by_dataset"):
        return
    rows = [{"dataset": name, **compute_testable_rate(model, ds, vars1, vars2)} for name, ds in datasets.items()]
    df = pd.DataFrame(rows)
    df["dataset"] = pd.Categorical(df["dataset"], categories=list(datasets.keys()), ordered=True)
    df = df.sort_values("dataset")
    height = 2.8 + 0.4 * max(0, len(df) - 2)
    fig, ax = plt.subplots(figsize=(7.2, height))
    sns.barplot(data=df, y="dataset", x="percent", ax=ax, color="#54A24B")
    ax.set_xlim(0, 100)
    label = "% of pairs that distinguish the hypotheses" if vars2 is not None else "% of pairs where patch changes output"
    ax.set_xlabel(label); ax.set_ylabel(""); ax.set_title(title)
    for i, row in enumerate(df.itertuples(index=False)):
        ax.text(float(row.percent) + 1, i, f"{float(row.percent):.1f}% ({int(row.testable)}/{int(row.total)})", va="center")
    plt.tight_layout(); plt.show()



from IPython.display import display
import textwrap
import re

def build_change_rate_table(datasets, variables):
    rows = []
    for name, ds in datasets.items():
        row = {"dataset": name, "n": len(ds)}
        for var in variables:
            if len(ds) == 0:
                row[f"Δ{var}"] = np.nan
            else:
                row[f"Δ{var}"] = 100 * sum(
                    ex["input"][var] != ex["counterfactual_inputs"][0][var]
                    for ex in ds
                ) / len(ds)
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["dataset"] = pd.Categorical(df["dataset"], categories=list(datasets.keys()), ordered=True)
        df = df.sort_values("dataset").reset_index(drop=True)
    return df

def display_change_rate_table(datasets, variables, *, title=None):
    df = build_change_rate_table(datasets, variables)
    pretty = df.copy()
    if "dataset" in pretty:
        pretty["dataset"] = pretty["dataset"].map(_pretty_inline_label)
    for var in variables:
        col = f"Δ{var}"
        if col in pretty:
            pretty[col] = pretty[col].map(lambda x: f"{x:.1f}%" if pd.notnull(x) else "—")
    pretty.columns = ["dataset", "n"] + [_pretty_inline_label(c) for c in pretty.columns[2:]]
    if title:
        print(title)
    display(pretty)
    return df

def plot_change_rate_heatmap(datasets, variables, *, title):
    if not _ensure_plotting("plot_change_rate_heatmap"):
        return
    df = build_change_rate_table(datasets, variables)
    if df.empty:
        print("(no data to plot)")
        return
    mat = df.set_index("dataset")[[f"Δ{var}" for var in variables]]
    pretty = mat.copy()
    pretty.index = [_pretty_wrap_label(idx, width=20) for idx in pretty.index]
    pretty.columns = [_pretty_wrap_label(col, width=18) for col in pretty.columns]
    fig, ax = plt.subplots(figsize=_heatmap_figsize(pretty, min_width=4.6, min_height=2.4))
    sns.heatmap(
        pretty,
        annot=True,
        fmt=".1f",
        annot_kws={"fontsize": _annot_fontsize(pretty), "fontweight": "semibold"},
        linewidths=0.6,
        linecolor="white",
        vmin=0,
        vmax=100,
        cmap="PuBuGn",
        cbar=True,
        cbar_kws={"label": "% of pairs", "orientation": "horizontal", "pad": 0.22, "shrink": 0.7, "aspect": 30},
        ax=ax,
    )
    _style_heatmap_axes(ax, x_rotation=0)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(title, pad=6, fontsize=10)
    fig.tight_layout()
    plt.show()

def build_effect_rate_table(model, datasets, effect_groups):
    items = list(effect_groups.items()) if isinstance(effect_groups, dict) else [
        ("+".join(vars), list(vars)) for vars in effect_groups
    ]
    rows = []
    for name, ds in datasets.items():
        row = {"dataset": name, "n": len(ds)}
        for label, vars_ in items:
            row[label] = compute_testable_rate(model, ds, list(vars_), None)["percent"]
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["dataset"] = pd.Categorical(df["dataset"], categories=list(datasets.keys()), ordered=True)
        df = df.sort_values("dataset").reset_index(drop=True)
    return df

def display_effect_rate_table(model, datasets, effect_groups, *, title=None):
    df = build_effect_rate_table(model, datasets, effect_groups)
    pretty = df.copy()
    if "dataset" in pretty:
        pretty["dataset"] = pretty["dataset"].map(_pretty_inline_label)
    for col in pretty.columns:
        if col not in {"dataset", "n"}:
            pretty[col] = pretty[col].map(lambda x: f"{x:.1f}%")
    pretty.columns = ["dataset", "n"] + [_pretty_inline_label(c) for c in pretty.columns[2:]]
    if title:
        print(title)
    display(pretty)
    return df

def plot_effect_rate_heatmap(model, datasets, effect_groups, *, title):
    if not _ensure_plotting("plot_effect_rate_heatmap"):
        return
    df = build_effect_rate_table(model, datasets, effect_groups)
    if df.empty:
        print("(no data to plot)")
        return
    effect_cols = [c for c in df.columns if c not in {"dataset", "n"}]
    mat = df.set_index("dataset")[effect_cols]
    pretty = mat.copy()
    pretty.index = [_pretty_wrap_label(idx, width=20) for idx in pretty.index]
    pretty.columns = [_pretty_metric_label(col, width=18) for col in pretty.columns]
    fig, ax = plt.subplots(figsize=_heatmap_figsize(pretty, min_width=4.8, min_height=2.4))
    sns.heatmap(
        pretty,
        annot=True,
        fmt=".1f",
        annot_kws={"fontsize": _annot_fontsize(pretty), "fontweight": "semibold"},
        linewidths=0.6,
        linecolor="white",
        vmin=0,
        vmax=100,
        cmap="magma",
        cbar=True,
        cbar_kws={"label": "% of pairs where output changes", "orientation": "horizontal", "pad": 0.24, "shrink": 0.7, "aspect": 30},
        ax=ax,
    )
    _style_heatmap_axes(ax, x_rotation=0)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title(title, pad=6, fontsize=10)
    fig.tight_layout()
    plt.show()

def _change_rate_for_vars(dataset, vars_):
    if len(dataset) == 0:
        return np.nan
    return 100 * sum(
        _any_value_changed(ex["input"], ex["counterfactual_inputs"][0], list(vars_))
        for ex in dataset
    ) / len(dataset)

def build_localization_scorecard(model, datasets, *, target_vars, control_vars=None):
    target_label = "+".join(target_vars)
    control_label = "+".join(control_vars) if control_vars else None
    rows = []
    for name, ds in datasets.items():
        row = {"dataset": name, "n": len(ds)}
        row[f"Δ target ({target_label})"] = _change_rate_for_vars(ds, target_vars)
        row[f"patch({target_label}) changes output"] = compute_testable_rate(model, ds, list(target_vars), None)["percent"]
        if control_vars:
            row[f"Δ control ({control_label})"] = _change_rate_for_vars(ds, control_vars)
            row[f"patch({control_label}) changes output"] = compute_testable_rate(model, ds, list(control_vars), None)["percent"]
            row[f"separates {target_label} vs {control_label}"] = compute_testable_rate(
                model, ds, list(target_vars), list(control_vars)
            )["percent"]
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["dataset"] = pd.Categorical(df["dataset"], categories=list(datasets.keys()), ordered=True)
        df = df.sort_values("dataset").reset_index(drop=True)
    return df

def display_localization_scorecard(model, datasets, *, target_vars, control_vars=None, title=None):
    df = build_localization_scorecard(model, datasets, target_vars=target_vars, control_vars=control_vars)
    pretty = df.copy()
    if "dataset" in pretty:
        pretty["dataset"] = pretty["dataset"].map(_pretty_inline_label)
    for col in pretty.columns:
        if col not in {"dataset", "n"}:
            pretty[col] = pretty[col].map(lambda x: f"{x:.1f}%" if pd.notnull(x) else "—")
    pretty.columns = ["dataset", "n"] + [_pretty_inline_label(c) for c in pretty.columns[2:]]
    if title:
        print(title)
    display(pretty)
    return df

def plot_localization_scorecard(model, datasets, *, target_vars, control_vars=None, title):
    if not _ensure_plotting("plot_localization_scorecard"):
        return
    df = build_localization_scorecard(model, datasets, target_vars=target_vars, control_vars=control_vars)
    if df.empty:
        print("(no data to plot)")
        return
    numeric_cols = [c for c in df.columns if c not in {"dataset", "n"}]
    mat = df.set_index("dataset")[numeric_cols]
    pretty = mat.copy()
    pretty.index = [_pretty_wrap_label(idx, width=18) for idx in pretty.index]
    pretty.columns = [_pretty_wrap_label(col, width=12) for col in pretty.columns]
    fig, ax = plt.subplots(figsize=_heatmap_figsize(pretty, min_width=6.0, min_height=2.8, max_width=9.5, max_height=4.2))
    sns.heatmap(
        pretty,
        annot=True,
        fmt=".1f",
        annot_kws={"fontsize": _annot_fontsize(pretty), "fontweight": "semibold"},
        linewidths=0.6,
        linecolor="white",
        vmin=0,
        vmax=100,
        cmap="BuPu",
        cbar=True,
        cbar_kws={"label": "% of pairs", "orientation": "horizontal", "pad": 0.24, "shrink": 0.7, "aspect": 30},
        ax=ax,
    )
    _style_heatmap_axes(ax, x_rotation=0)
    ax.set_xlabel("")  # column headers are already self-describing
    ax.set_ylabel("")
    ax.set_title(title, pad=6, fontsize=10)
    fig.tight_layout()
    plt.show()

def plot_dataset_criteria_bars(model, datasets, *, target_vars, control_vars=None, title=None, figsize=None):
    """Compact horizontal grouped bar chart summarizing the three criteria across all datasets.

    One row per diagnostic (criterion-1 for V, criterion-2 for V, criterion-1 for W,
    criterion-2 for W, criterion-3 distinguishability), three bars per row (one per
    dataset). Replaces the three separate heatmaps (change-rate, effect-rate, scorecard)
    with a single, information-dense view where it is easy to scan "which dataset best
    satisfies which criterion".

    Criterion definitions (exact formulas)
    --------------------------------------
    Let pair = (base b, source s) and let H be the causal model.

    - (1) for V: fraction of pairs with b[V] != s[V].
    - (2) for V: fraction of pairs with H(b | V:=s[V]).raw_output != b.raw_output.
    - (1) for W: fraction of pairs with b[W] != s[W] (diagnostic only).
    - (2) for W: fraction of pairs with H(b | W:=s[W]).raw_output != b.raw_output (diagnostic only).
    - (3) V vs W: fraction of pairs with H(b | V:=s[V]).raw_output != H(b | W:=s[W]).raw_output.
    """
    if not _ensure_plotting("plot_dataset_criteria_bars"):
        return
    df = build_localization_scorecard(model, datasets, target_vars=target_vars, control_vars=control_vars)
    if df.empty:
        print("(no data to plot)")
        return

    target_label = "+".join(target_vars)
    control_label = "+".join(control_vars) if control_vars else None

    # Human-readable criterion rows, in a fixed pedagogical order.
    row_specs: list[tuple[str, str]] = [
        (f"Δ target ({target_label})", f"(1) Δ {target_label}"),
        (f"patch({target_label}) changes output", f"(2) patch({target_label}) → ΔY"),
    ]
    if control_label:
        row_specs += [
            (f"Δ control ({control_label})", f"(1') Δ {control_label}  (diag.)"),
            (f"patch({control_label}) changes output", f"(2') patch({control_label}) → ΔY  (diag.)"),
            (f"separates {target_label} vs {control_label}", f"(3) Y_V(b,s) ≠ Y_W(b,s)"),
        ]

    long = df.melt(id_vars=["dataset"], value_vars=[spec[0] for spec in row_specs],
                   var_name="criterion_raw", value_name="pct")
    rename_map = {src: dst for src, dst in row_specs}
    long["criterion"] = long["criterion_raw"].map(rename_map)
    long["criterion"] = pd.Categorical(long["criterion"], categories=[dst for _, dst in row_specs], ordered=True)
    long["dataset"] = long["dataset"].astype(str).map(_pretty_inline_label)
    long["dataset"] = pd.Categorical(long["dataset"], categories=[_pretty_inline_label(n) for n in datasets.keys()], ordered=True)

    # Compact sizing: height scales with number of criterion rows × datasets.
    n_rows = len(row_specs)
    n_ds = len(datasets)
    default_height = min(4.4, max(2.2, 0.42 * n_rows * n_ds + 0.8))
    if figsize is None:
        figsize = (7.2, default_height)

    fig, ax = plt.subplots(figsize=figsize)
    palette = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"][:n_ds]
    sns.barplot(
        data=long, y="criterion", x="pct", hue="dataset",
        orient="h", ax=ax, palette=palette, saturation=1.0, edgecolor="none",
    )
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of pairs", fontsize=9, labelpad=4)
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelsize=8, pad=2)
    ax.tick_params(axis="y", labelsize=9, pad=2)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, pad=6, fontsize=10)

    # Annotate each bar with its percentage value.
    for bar in ax.patches:
        if not hasattr(bar, "get_width"):
            continue
        width = bar.get_width()
        if width is None or np.isnan(width):
            continue
        y = bar.get_y() + bar.get_height() / 2
        ax.text(float(width) + 1.2, y, f"{float(width):.0f}%", va="center", ha="left", fontsize=7.5, color="#333")

    ax.legend(title="dataset", loc="upper left", bbox_to_anchor=(1.02, 1.0),
              fontsize=8, title_fontsize=8, frameon=False, borderaxespad=0.0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    plt.show()
    return df


def _short_text(value, width=84):
    text = str(value).replace("\n", " ")
    return textwrap.shorten(text, width=width, placeholder="…")

def preview_counterfactual_pairs(dataset, variables, *, n=2, title=None, width=84):
    rows = []
    for ex in dataset[:n]:
        base = ex["input"]
        cf = ex["counterfactual_inputs"][0]
        row = {}
        if "raw_input" in base:
            row["base input"] = _short_text(base["raw_input"], width=width)
            row["cf input"] = _short_text(cf["raw_input"], width=width)
        for var in variables:
            row[f"base {var}"] = base[var]
            row[f"cf {var}"] = cf[var]
            row[f"Δ{var}"] = bool(base[var] != cf[var])
        if "raw_output" in base:
            row["base out"] = str(base["raw_output"]).strip()
            row["cf out"] = str(cf["raw_output"]).strip()
        rows.append(row)
    df = pd.DataFrame(rows)
    if title:
        print(title)
    display(df)
    return df

# ─── Shared activation patching helpers ─────────────────────────────────────

import os
from typing import Any
from causalab.experiments.filter import filter_dataset
from causalab.experiments.interchange_targets import (
    build_residual_stream_targets, detect_component_type_from_targets, extract_grid_dimensions_from_targets)
from causalab.experiments.jobs.interchange_score_grid import run_interchange_score_heatmap
from causalab.causal.causal_utils import save_counterfactual_examples
from causalab.neural.pipeline import LMPipeline

def default_layers(pipeline, device):
    num_layers = pipeline.model.config.num_hidden_layers
    layers = [-1] + list(range(num_layers))
    return layers


def run_patching_for_datasets(*, pipeline, causal_model, datasets, token_positions, target_variable_groups, metric, batch_size, layers, output_root=None):
    if output_root is None:
        import tempfile
        output_root = tempfile.mkdtemp(prefix="causalab_patch_")
    os.makedirs(output_root, exist_ok=True)
    targets = build_residual_stream_targets(pipeline=pipeline, layers=layers, token_positions=token_positions, mode="one_target_per_unit")
    results = {}
    for name, dataset in datasets.items():
        if len(dataset) == 0:
            print(f"  (skip) dataset '{name}' is empty"); continue
        dataset_path = os.path.join(output_root, f"{name}.json")
        save_counterfactual_examples(dataset, dataset_path)
        print(f"\nRunning patching on '{name}' ({len(dataset)} examples)")
        result = run_interchange_score_heatmap(
            causal_model=causal_model, interchange_targets=targets, dataset_path=dataset_path, pipeline=pipeline,
            target_variable_groups=target_variable_groups, batch_size=batch_size,
            output_dir=os.path.join(output_root, f"results_{name}"), metric=metric, save_results=False, verbose=True)
        results[name] = result["scores"]
    return results, targets

def scores_to_matrix(scores_dict, targets):
    comp_type = detect_component_type_from_targets(targets)
    dims = extract_grid_dimensions_from_targets(comp_type, targets)
    layers = dims["layers"]; pos_ids = dims["token_position_ids"]
    mat = np.full((len(layers), len(pos_ids)), np.nan)
    for (layer, pos_id), score in scores_dict.items():
        if layer in layers and pos_id in pos_ids:
            mat[layers.index(layer), pos_ids.index(pos_id)] = float(score)
    return mat, pos_ids, layers

def _plot_trace_style_matrix(
    ax,
    mat,
    x_labels,
    y_labels,
    title,
    *,
    cmap="Purples",
    vmin=0.0,
    vmax=1.0,
    annotate=True,
    value_fmt=".2f",
):
    display_x = [_pretty_wrap_label(lbl, width=12) for lbl in x_labels]
    im = ax.imshow(mat, interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("position", labelpad=8)
    ax.set_ylabel("layer", labelpad=8)
    ax.set_xticks(np.arange(len(display_x)))
    ax.set_xticklabels(display_x, rotation=0)

    tick_positions = np.arange(0, len(y_labels), 2) if len(y_labels) > 8 else np.arange(len(y_labels))
    ax.set_yticks(tick_positions)
    ax.set_yticklabels([y_labels[i] for i in tick_positions])
    ax.tick_params(axis="both", which="both", length=0, labelsize=9, top=False, labeltop=False, bottom=True, labelbottom=True, pad=4)
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)

    if annotate:
        threshold = (vmin + vmax) / 2.0
        fontsize = _iia_annot_fontsize(mat)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                value = mat[i, j]
                if np.isnan(value):
                    continue
                text_color = "white" if value >= threshold else "black"
                # Bold only "strong" cells so the eye is drawn to them; use
                # |value| > 0.5 so the rule is meaningful both for absolute-IIA
                # panels (values in [0, 1]) and for the ΔIIA panel (values in [-1, 1]).
                is_strong = float(abs(value)) > 0.5
                weight = "bold" if is_strong else "normal"
                ax.text(
                    j, i, format(value, value_fmt),
                    ha="center", va="center",
                    color=text_color, fontsize=fontsize, fontweight=weight,
                )
    return im

def plot_iia_triplet(*, results, targets, var_name, dataset_good, dataset_bad, pos_id_to_label=None, title_prefix=""):
    if not _ensure_plotting("plot_iia_triplet"):
        return
    key = (var_name,)
    if dataset_good not in results or dataset_bad not in results:
        print(f"(skip) missing scores for {dataset_good} or {dataset_bad}")
        return
    if key not in results[dataset_good] or key not in results[dataset_bad]:
        print(f"(skip) missing {key} scores")
        return

    scores_good = results[dataset_good][key]
    scores_bad = results[dataset_bad][key]
    mat_good, pos_ids, layers = scores_to_matrix(scores_good, targets)
    mat_bad, _, _ = scores_to_matrix(scores_bad, targets)

    y_labels = [f"L{layer}" if layer >= 0 else "Embed" for layer in layers]
    mat_good = np.flipud(mat_good)
    mat_bad = np.flipud(mat_bad)
    diff = mat_good - mat_bad
    y_labels = list(reversed(y_labels))

    x_labels = [pos_id_to_label.get(p, str(p).replace("_", "\n")) for p in pos_ids] if pos_id_to_label else [str(p).replace("_", "\n") for p in pos_ids]
    annotate = True

    abs_vmin, abs_vmax = 0.0, 1.0

    diff_max = max(0.05, float(np.nanmax(np.abs(diff))))

    panel_width = max(4.6, 0.78 * len(x_labels) + 1.8)
    fig_width = 3 * panel_width + 1.0
    fig_height = max(5.2, 0.35 * len(y_labels) + 3.2)

    fig, axes = plt.subplots(1, 3, figsize=(fig_width, fig_height), constrained_layout=True)
    im0 = _plot_trace_style_matrix(
        axes[0],
        mat_bad,
        x_labels,
        y_labels,
        f"{title_prefix}{_pretty_inline_label(dataset_bad)}\npatch({_pretty_inline_label(var_name)})",
        cmap="Purples",
        vmin=abs_vmin,
        vmax=abs_vmax,
        annotate=annotate,
    )
    im1 = _plot_trace_style_matrix(
        axes[1],
        mat_good,
        x_labels,
        y_labels,
        f"{title_prefix}{_pretty_inline_label(dataset_good)}\npatch({_pretty_inline_label(var_name)})",
        cmap="Purples",
        vmin=abs_vmin,
        vmax=abs_vmax,
        annotate=annotate,
    )
    im2 = _plot_trace_style_matrix(
        axes[2],
        diff,
        x_labels,
        y_labels,
        f"{title_prefix}ΔIIA\n(isolated − comparison): {_pretty_inline_label(var_name)}",
        cmap="RdBu_r",
        vmin=-diff_max,
        vmax=diff_max,
        annotate=annotate,
        value_fmt="+.2f",
    )

    cbar0 = fig.colorbar(im1, ax=axes[:2], shrink=0.82, location="bottom", pad=0.12, aspect=40)
    cbar0.set_label("IIA", fontsize=10)
    cbar2 = fig.colorbar(im2, ax=axes[2], shrink=0.82, location="bottom", pad=0.12, aspect=40)
    cbar2.set_label("ΔIIA", fontsize=10)

    fig.text(
        0.5,
        -0.03,
        "Darker purple means higher IIA. In the difference panel, red means the isolated dataset is higher; blue means the comparison dataset is higher.",
        ha="center",
        fontsize=9,
    )
    plt.show()

def plot_iia_target_control_random(*, results, targets, var_name, dataset_targeted, dataset_control, dataset_random=None, pos_id_to_label=None, title_prefix=""):
    # Clean, matched comparison
    plot_iia_triplet(
        results=results,
        targets=targets,
        var_name=var_name,
        dataset_good=dataset_targeted,
        dataset_bad=dataset_control,
        pos_id_to_label=pos_id_to_label,
        title_prefix=title_prefix,
    )

    # Messier random contrast
    if dataset_random is None:
        return
    if dataset_random not in results or (var_name,) not in results.get(dataset_random, {}):
        print(f"(skip) random comparison unavailable for {dataset_random} / {var_name}")
        return
    plot_iia_triplet(
        results=results,
        targets=targets,
        var_name=var_name,
        dataset_good=dataset_targeted,
        dataset_bad=dataset_random,
        pos_id_to_label=pos_id_to_label,
        title_prefix=title_prefix,
    )



# ─── Pair printing helpers (text-only) ──────────────────────────────────────

def print_counterfactual_pair_text(
    example,
    *,
    variables,
    title,
    wrap: int = 96,
):
    """Readable text-only rendering of one base/source pair."""
    base = example["input"]
    cf = example["counterfactual_inputs"][0]

    print(f"\n--- {title} ---")

    if "raw_input" in base:
        print("base input:")
        print("  " + textwrap.fill(str(base["raw_input"]).strip(), width=wrap, subsequent_indent="  "))
    if "raw_input" in cf:
        print("source input:")
        print("  " + textwrap.fill(str(cf["raw_input"]).strip(), width=wrap, subsequent_indent="  "))

    print("variable values:")
    for v in variables:
        b = base.get(v)
        s = cf.get(v)
        delta = "CHANGED" if b != s else "same"
        print(f"  - {v}: base={b} | source={s} | {delta}")

    if "raw_output" in base or "raw_output" in cf:
        print("outputs:")
        if "raw_output" in base:
            print(f"  base output:   {str(base['raw_output']).strip()}")
        if "raw_output" in cf:
            print(f"  source output: {str(cf['raw_output']).strip()}")


def _patch_outputs(model, ex, *, target, other):
    base = ex["input"]
    cf = ex["counterfactual_inputs"][0]
    y_t = str(model.run_interchange(base, {target: cf})["raw_output"]).strip()
    y_o = str(model.run_interchange(base, {other: cf})["raw_output"]).strip()
    return y_t, y_o


def print_one_pair_per_dataset(
    datasets,
    *,
    causal_model,
    variables,
    target,
    other,
    title,
    wrap: int = 100,
):
    """For each dataset, print one (base, source) pair and the two patch outcomes."""
    print(f"\n=== {title} ===")
    for name, ds in datasets.items():
        if not ds:
            print(f"\n--- {name} (empty) ---"); continue
        ex = ds[0]
        print_counterfactual_pair_text(
            ex, variables=variables,
            title=f"{name.replace('_', ' ')} dataset — one example pair",
            wrap=wrap,
        )
        y_t, y_o = _patch_outputs(causal_model, ex, target=target, other=other)
        print(f"  patch({target}) -> {y_t}")
        print(f"  patch({other})  -> {y_o}")


def print_confounding_and_nonconfounding(
    *,
    causal_model,
    variables,
    target,
    other,
    confounding_source,
    nonconfounding_source,
    title,
    confounding_filter=None,
    nonconfounding_filter=None,
    wrap: int = 100,
):
    """Pick one confounding pair (patch(V) and patch(W) yield the same output) and one
    non-confounding pair (they differ) and print them side by side.

    ``confounding_source`` / ``nonconfounding_source`` are lists of examples to search
    through; ``*_filter`` are optional extra predicates on each example.
    """
    def _is_confounding(ex):
        y_t, y_o = _patch_outputs(causal_model, ex, target=target, other=other)
        return y_t == y_o

    conf = next(
        (
            ex for ex in confounding_source
            if _is_confounding(ex) and (confounding_filter is None or confounding_filter(ex))
        ),
        None,
    )
    nonconf = next(
        (
            ex for ex in nonconfounding_source
            if (not _is_confounding(ex)) and (nonconfounding_filter is None or nonconfounding_filter(ex))
        ),
        None,
    )

    print(f"\n=== {title} ===")
    for label, ex in [
        ("confounding example", conf),
        ("non-confounding example", nonconf),
    ]:
        if ex is None:
            print(f"  (warning) could not find {label}"); continue
        print_counterfactual_pair_text(ex, variables=variables, title=label, wrap=wrap)
        y_t, y_o = _patch_outputs(causal_model, ex, target=target, other=other)
        print(f"  patch({target}) -> {y_t}")
        print(f"  patch({other})  -> {y_o}")
        verdict = (
            f"CONFOUNDING: patch({target}) and patch({other}) yield the same output -> indistinguishable"
            if y_t == y_o else
            f"NON-CONFOUNDING: patch({target}) and patch({other}) yield different outputs -> distinguishable"
        )
        print(f"  interpretation: {verdict}")


# ─── Counterfactual dataset builder (target/isolated/random) ────────────────

def _satisfies_criteria(
    example, *, causal_model, target, other,
    require_distinguishability, require_other_fixed,
):
    base = example["input"]
    cf = example["counterfactual_inputs"][0]

    if base[target] == cf[target]:
        return False
    y_target = causal_model.run_interchange(base, {target: cf})["raw_output"]
    if y_target == base["raw_output"]:
        return False
    if require_other_fixed and base[other] != cf[other]:
        return False
    if require_distinguishability:
        y_other = causal_model.run_interchange(base, {other: cf})["raw_output"]
        if y_target == y_other:
            return False
    return True


def build_counterfactual_dataset(
    *,
    causal_model,
    sampler: Callable[[], "CounterfactualExample"],
    target: str,
    other: str,
    n: int,
    require_distinguishability: bool = False,
    require_other_fixed: bool = False,
    max_attempts_per_example: int = 300,
):
    """Draw ``n`` counterfactual pairs that satisfy the selected criteria.

    Criteria (see Section 0 of the lecture notebook):

    - (1) variable-level change for the target: ``base[target] != source[target]``.
    - (2) test-counterfactual: patching ``target`` into ``base`` changes the output.
    - (3) distinguishability against ``other``: patch(target) != patch(other).

    ``require_distinguishability`` enforces (3); ``require_other_fixed`` additionally
    pins ``other`` to the same base value (useful for the "isolated" dataset).
    """
    out = []
    max_attempts = max_attempts_per_example * n
    for _ in range(max_attempts):
        if len(out) >= n:
            break
        ex = sampler()
        if _satisfies_criteria(
            ex,
            causal_model=causal_model, target=target, other=other,
            require_distinguishability=require_distinguishability,
            require_other_fixed=require_other_fixed,
        ):
            out.append(ex)
    if len(out) < n:
        raise RuntimeError(
            f"Could only build {len(out)}/{n} examples for target={target}, "
            f"require_distinguishability={require_distinguishability}."
        )
    return out


# ─── Pipeline loaders and filter/report ─────────────────────────────────────

def load_activation_patching_pipeline(
    *,
    model_candidates=None,
    max_new_tokens: int = 5,
    max_length: int = 768,
):
    """Pick the first model in ``model_candidates`` that loads on the current device.

    Returns ``(pipeline, device_str)``. The ``CAUSALAB_PATCHING_MODEL`` env var,
    if set, is tried first. This is the shared pipeline used by Parts 1 and 2
    (hierarchical equality, MCQA).
    """
    import os
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    defaults = [
        os.environ.get("CAUSALAB_PATCHING_MODEL"),
        "meta-llama/Llama-3.2-1B-Instruct",
        "Qwen/Qwen2.5-0.5B-Instruct",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    ]
    candidates = [m for m in (model_candidates or defaults) if m]

    pipeline, last_err, picked = None, None, None
    for name in candidates:
        try:
            pipeline = LMPipeline(
                name,
                max_new_tokens=max_new_tokens,
                device=device,
                dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                max_length=max_length,
            )
            picked = name
            break
        except Exception as e:
            last_err = e
    if pipeline is None:
        raise RuntimeError("Failed to load any model for activation patching") from last_err
    print(f"Activation-patching model: {picked} | device: {device}")
    return pipeline, device


def load_entity_binding_pipeline(
    *,
    device: str | None = None,
    max_new_tokens: int = 5,
    max_length: int = 256,
):
    """Dedicated GPT-2 pipeline for Part 3.

    The entity-binding structured token-position parser in
    ``causalab.tasks.entity_binding.token_positions`` was developed and tested
    against GPT-2's tokenization of the canonical ``love`` prompt format, so we
    use GPT-2 here instead of the larger pipeline shared by Parts 1 and 2.
    """
    import torch
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = LMPipeline(
        "gpt2",
        max_new_tokens=max_new_tokens,
        device=device,
        dtype=torch.float32,
        max_length=max_length,
    )
    print(f"Entity-binding pipeline: gpt2 | device: {device}")
    return pipeline


def filter_and_report(
    *,
    datasets,
    pipeline,
    causal_model,
    metric,
    batch_size: int,
    label: str = "",
    min_kept_warn: int = 4,
):
    """Run ``filter_dataset`` on each entry of ``datasets`` and report kept counts.

    Raises ``RuntimeError`` if any dataset has zero surviving examples; this
    almost always indicates a bug in the sampler or prompt template, and
    silently falling back to unfiltered examples would hide that bug.
    """
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}Filtering datasets (keep only examples the LM gets right)...")
    kept_by_name = {}
    for name, ds in datasets.items():
        kept = filter_dataset(
            dataset=ds,
            pipeline=pipeline,
            causal_model=causal_model,
            metric=metric,
            batch_size=batch_size,
        )
        print(f"  {name}: kept {len(kept)}/{len(ds)} examples")
        if len(kept) == 0:
            raise RuntimeError(
                f"No examples survived filtering for '{name}'. "
                "Check the sampler distribution against the prompt template; "
                "the LM is unable to answer any of the generated inputs."
            )
        if len(kept) < min_kept_warn:
            print("    warning: very few examples survived; heatmaps may be noisy.")
        kept_by_name[name] = kept
    return kept_by_name


# ─── Task-specific metrics and token-position panels ───────────────────────

def hier_metric(neural_output, causal_output):
    """Hierarchical-equality filter: does the LM's first word match 'Yes' / 'No'?"""
    expected = causal_output.strip().lower()
    first_word = neural_output.get("string", "").strip().lower().split()
    return bool(first_word) and first_word[0] == expected


def mcqa_metric(neural_output, causal_output):
    """MCQA filter: does the LM produce the correct answer letter (A, B, ...)?"""
    expected = causal_output.strip().upper()
    text = neural_output.get("string", "").upper()
    letters = re.findall(r"\b([A-Z])\b", text)
    if letters:
        return letters[0] == expected
    return text.strip()[:1] == expected


def build_hier_token_positions(pipeline, *, prompt_template):
    """Build the (positions, label_map) panel used by the hierarchical-equality task."""
    from causalab.neural.token_position_builder import build_token_position_factories

    specs = {
        "left_a":     {"type": "index", "position": 0, "scope": {"variable": "left_a"}},
        "left_b":     {"type": "index", "position": 0, "scope": {"variable": "left_b"}},
        "right_a":    {"type": "index", "position": 0, "scope": {"variable": "right_a"}},
        "right_b":    {"type": "index", "position": 0, "scope": {"variable": "right_b"}},
        "last_token": {"type": "index", "position": -1},
    }
    factories = build_token_position_factories(specs, prompt_template)
    order = ["left_a", "left_b", "right_a", "right_b", "last_token"]
    positions = [factories[name](pipeline) for name in order]
    label_map = {tp.id: name.replace("_", "\n") for name, tp in zip(order, positions)}
    return {"positions": positions, "label_map": label_map}


def build_mcqa_token_positions(pipeline):
    """Build the (positions, label_map) panel used by the MCQA task."""
    from causalab.tasks.MCQA.token_positions import create_token_positions

    all_positions = create_token_positions(pipeline)
    order = [
        "correct_symbol", "correct_symbol_period",
        "symbol0", "symbol0_period", "symbol1", "symbol1_period", "last_token",
    ]
    positions = [all_positions[name] for name in order]
    label_aliases = {
        "correct_symbol": "correct\nsymbol", "correct_symbol_period": "correct\nsymbol.",
        "symbol0": "choice\n0", "symbol0_period": "choice\n0.",
        "symbol1": "choice\n1", "symbol1_period": "choice\n1.",
        "last_token": "last\ntoken",
    }
    label_map = {tp.id: label_aliases[name] for name, tp in zip(order, positions)}
    return {"positions": positions, "label_map": label_map}


# ─── Entity-binding task helpers ────────────────────────────────────────────

def install_entity_binding_legacy_adapter():
    """Monkey-patch the entity-binding task factories to accept the current
    ``CausalModel`` signature.

    The attached entity-binding task code uses an older ``CausalModel(...)``
    constructor. The adapter below takes ``(variables, values, parents,
    mechanisms)`` (the old signature) and builds the upgraded ``CausalModel``.
    Safe to call repeatedly.
    """
    import causalab.tasks.entity_binding.causal_models as _eb_cm

    def _adapter(variables, values, parents, mechanisms, *, id: str = "null", print_pos=None):
        upgraded = {}
        for var in variables:
            par = list(parents.get(var, []))
            fn = mechanisms.get(var, (lambda t: None))
            if len(par) == 0 and isinstance(values.get(var, None), list):
                upgraded[var] = input_var(list(values[var]))
            else:
                upgraded[var] = Mechanism(parents=par, compute=fn)
        return CausalModel(upgraded, values, print_pos=print_pos, id=id)

    _eb_cm.CausalModel = _adapter


def install_entity_binding_parser_patch():
    """Fix the pinned ``PromptParser.parse_prompt`` so it respects
    ``prompt_prefix`` and multi-character statement/question separators.

    The pinned causalab release returns entity character offsets in
    statement-local coordinates. The downstream tokenizer then maps those onto
    the full tokenized prompt (which does include the prefix), so entity-0 in
    the statement resolves to a token inside the prompt prefix ('will',
    'question', 'about', ...). This patch shifts the offsets appropriately. It
    activates only when the installed version lacks the fix.
    """
    import causalab.tasks.entity_binding.token_positions as _eb_tp
    import inspect as _inspect

    src = _inspect.getsource(_eb_tp.PromptParser.parse_prompt)
    if "statement_start" in src:
        return  # already fixed upstream

    _orig = _eb_tp.PromptParser.parse_prompt

    def _patched(self, input_sample):
        parsed = _orig(self, input_sample)
        raw_text = parsed.raw_text or ""
        prefix = getattr(self.config, "prompt_prefix", "") or ""
        separator = getattr(self.config, "statement_question_separator", " ") or " "

        if prefix and raw_text.startswith(prefix):
            statement_start = len(prefix)
        elif parsed.statement_region is not None:
            stmt_len = parsed.statement_region[1] - parsed.statement_region[0]
            stmt_text = raw_text[:stmt_len] if stmt_len <= len(raw_text) else raw_text
            idx = raw_text.find(stmt_text) if stmt_text else -1
            statement_start = idx if idx > 0 else 0
        else:
            statement_start = 0

        q_shift = statement_start + (len(separator) - 1)
        old_stmt = parsed.statement_region
        old_q = parsed.question_region
        for seg in parsed.segments:
            if old_stmt is not None and old_stmt[0] <= seg.char_start < old_stmt[1]:
                seg.char_start += statement_start
                seg.char_end += statement_start
            elif old_q is not None and old_q[0] <= seg.char_start < old_q[1]:
                seg.char_start += q_shift
                seg.char_end += q_shift
        if old_stmt is not None:
            parsed.statement_region = (old_stmt[0] + statement_start, old_stmt[1] + statement_start)
        if old_q is not None:
            parsed.question_region = (old_q[0] + q_shift, old_q[1] + q_shift)
        return parsed

    _eb_tp.PromptParser.parse_prompt = _patched


def install_entity_binding_shims():
    """Convenience: apply both entity-binding shims in one call."""
    install_entity_binding_legacy_adapter()
    install_entity_binding_parser_patch()


def make_entity_binding_generators(config, positional_model):
    """Build the two task-local counterfactual generators for Part 3.

    Returns ``{"query_swap": ..., "random": ...}``. Both generators return
    pre-materialized traces so the rest of the pipeline (filter, patching,
    scorecards) sees the same shape as Parts 1 and 2.

    - ``query_swap``: change which entity is named in the question, keep the
      recovered group fixed; so Δquery_entity = 1, Δpositional_query_group = 0.
    - ``random``: reuse the task-code random generator, keeping only examples
      where all ``config.max_groups`` groups are filled. Intentionally messy.
    """
    from causalab.tasks.entity_binding.counterfactuals import (
        random_counterfactual as entity_random_counterfactual,
    )
    from causalab.tasks.entity_binding.causal_models import sample_valid_entity_binding_input

    def _to_plain_dict(sample):
        if hasattr(sample, "to_dict"):
            sample = sample.to_dict()
        return dict(sample)

    def _clean_inputs(sample):
        sample = _to_plain_dict(sample)
        for key in ["raw_input", "raw_output", "query_entity", "positional_query_group"]:
            sample.pop(key, None)
        if "query_indices" in sample:
            sample["query_indices"] = tuple(sample["query_indices"])
        sample.setdefault("statement_template", config.statement_template)
        return sample

    def _materialize(sample):
        return positional_model.new_trace(_clean_inputs(sample))

    def _materialize_example(example):
        return {
            "input": _materialize(example["input"]),
            "counterfactual_inputs": [_materialize(example["counterfactual_inputs"][0])],
        }

    def _sample_base():
        while True:
            candidate = _clean_inputs(sample_valid_entity_binding_input(config))
            if candidate["active_groups"] == config.max_groups:
                return candidate

    def query_swap():
        base = _sample_base()
        qg = base["query_group"]
        qidxs = tuple(base["query_indices"])
        other_groups = [g for g in range(base["active_groups"]) if g != qg]
        swap_group = random.choice(other_groups)
        cf = dict(base)
        for qidx in qidxs:
            k_q = f"entity_g{qg}_e{qidx}"
            k_swap = f"entity_g{swap_group}_e{qidx}"
            cf[k_q], cf[k_swap] = base[k_swap], base[k_q]
        return _materialize_example({"input": base, "counterfactual_inputs": [cf]})

    def random_pair():
        while True:
            ex = entity_random_counterfactual(config)
            base = _clean_inputs(ex["input"])
            cf = _clean_inputs(ex["counterfactual_inputs"][0])
            if (
                base["active_groups"] == config.max_groups
                and cf["active_groups"] == config.max_groups
            ):
                return _materialize_example({"input": base, "counterfactual_inputs": [cf]})

    return {"query_swap": query_swap, "random": random_pair}


def build_entity_binding_token_positions(config, pipeline):
    """Build a richer panel of entity-binding intervention sites than the
    shipped task plugin exposes.

    We need more than the canonical ``last_token`` site because that site
    conflates representations of ``query_entity`` and ``positional_query_group``.
    The richer panel covers:

    1. Per-entity statement sites ``g{g}_e{e}_stmt``: last token of each entity
       span in the statement region.
    2. ``query_entity_q``: last token of the queried entity inside the question
       region.
    3. ``last_token``: final prompt token (canonical reference site).
    4. ``both_e1_swap``: the ``both_e1`` site from the reference code, which
       returns the last tokens of e1 in groups 0 and 1 with the order reversed
       on the counterfactual run so that patching interchanges the two
       representations simultaneously.

    Returns a dict with keys::

        {
            "positions": list[TokenPosition],          # in display order
            "positions_dict": dict[str, TokenPosition],
            "position_order": list[str],
            "label_map": dict[str, str],               # position id -> display label
            "resolver_errors": list[str],              # populated during indexing
        }
    """
    from causalab.neural.token_position_builder import TokenPosition, get_last_token_index
    from causalab.tasks.entity_binding.token_positions import (
        get_entity_token_indices_structured,
    )

    resolver_errors = []

    def _normalize(inp):
        if hasattr(inp, "to_dict"):
            try:
                d = dict(inp.to_dict())
            except Exception:
                d = {}
            if not d.get("raw_input"):
                try:
                    d["raw_input"] = inp["raw_input"]
                except Exception:
                    pass
        else:
            d = dict(inp)
        if "query_indices" in d and isinstance(d["query_indices"], list):
            d["query_indices"] = tuple(d["query_indices"])
        d.setdefault("statement_template", config.statement_template)
        return d

    def _resolve(inp, *, group_idx, entity_idx, region):
        try:
            tokens = get_entity_token_indices_structured(
                _normalize(inp), pipeline, config,
                group_idx=group_idx, entity_idx=entity_idx, region=region,
            )
        except (ValueError, KeyError, TypeError) as exc:
            tag = f"g{group_idx}_e{entity_idx}_{region}"
            if not any(tag in msg for msg in resolver_errors):
                resolver_errors.append(f"[{tag}] {type(exc).__name__}: {exc}")
            return None
        return tokens or None

    def _make_entity_indexer(*, group_idx, entity_idx, region):
        def _indexer(inp, is_original: bool = True):
            toks = _resolve(inp, group_idx=group_idx, entity_idx=entity_idx, region=region)
            if toks:
                return [toks[-1]]
            return get_last_token_index(inp, pipeline)
        return _indexer

    positions_dict = {}
    entities_per_group = getattr(config, "max_entities_per_group", 2)

    for g in range(config.max_groups):
        for e in range(entities_per_group):
            site_id = f"g{g}_e{e}_stmt"
            positions_dict[site_id] = TokenPosition(
                _make_entity_indexer(group_idx=g, entity_idx=e, region="statement"),
                pipeline, id=site_id,
            )

    def _query_entity_indexer(inp, is_original: bool = True):
        try:
            d = _normalize(inp)
            qidx = d["query_indices"][0]
            qgroup = int(d["query_group"])
            toks = get_entity_token_indices_structured(
                d, pipeline, config,
                group_idx=qgroup, entity_idx=qidx, region="question",
            )
        except (ValueError, KeyError, TypeError):
            toks = None
        if toks:
            return [toks[-1]]
        return get_last_token_index(inp, pipeline)

    positions_dict["query_entity_q"] = TokenPosition(
        _query_entity_indexer, pipeline, id="query_entity_q",
    )
    positions_dict["last_token"] = TokenPosition(
        lambda inp, is_original=True: get_last_token_index(inp, pipeline),
        pipeline, id="last_token",
    )

    def _both_e1_indexer(inp, is_original: bool = True):
        d = _normalize(inp)
        tokens_list = []
        for gid in (0, 1):
            toks = _resolve(d, group_idx=gid, entity_idx=1, region="statement")
            tokens_list.append(toks[-1] if toks else None)
        if not is_original:
            tokens_list = tokens_list[::-1]
        resolved = [t for t in tokens_list if t is not None]
        if resolved:
            return resolved
        return get_last_token_index(inp, pipeline)

    if config.max_groups >= 2:
        positions_dict["both_e1_swap"] = TokenPosition(
            _both_e1_indexer, pipeline, is_original=True, id="both_e1_swap",
        )

    position_order = [
        f"g{g}_e{e}_stmt"
        for g in range(config.max_groups)
        for e in range(entities_per_group)
    ]
    if "both_e1_swap" in positions_dict:
        position_order.append("both_e1_swap")
    position_order += ["query_entity_q", "last_token"]

    label_map = {
        "query_entity_q": "query\nentity\n(Q)",
        "last_token": "last\ntoken",
        "both_e1_swap": "both\ne1\nswap",
    }
    for g in range(config.max_groups):
        for e in range(entities_per_group):
            label_map[f"g{g}_e{e}_stmt"] = f"g{g}\ne{e}\n(S)"

    positions = [positions_dict[name] for name in position_order]
    pos_id_to_label = {tp.id: label_map[name] for name, tp in zip(position_order, positions)}

    return {
        "positions": positions,
        "positions_dict": positions_dict,
        "position_order": position_order,
        "label_map": pos_id_to_label,
        "resolver_errors": resolver_errors,
    }


def diagnose_entity_binding_sites(*, filtered_datasets, pipeline, config, site_panel):
    """Resolve every site on one filtered example, print the result, and hard-guard
    against silent fallbacks or prefix-region hits.

    ``site_panel`` is the dict returned by ``build_entity_binding_token_positions``.
    Raises ``RuntimeError`` when either the structured parser fell back to
    ``last_token`` on any site, or when some statement-entity site lands inside
    the prompt prefix (which would otherwise produce a coherent-looking grid of
    prefix tokens like 'will', 'question', 'about').
    """
    positions_dict = site_panel["positions_dict"]
    position_order = site_panel["position_order"]
    resolver_errors = site_panel["resolver_errors"]

    diag_dataset = next((d for d in filtered_datasets.values() if len(d) > 0), None)
    print("\nIntervention sites resolved on one filtered base example:")
    if not diag_dataset:
        print("  (no filtered example available)")
        return

    diag_inp = diag_dataset[0]["input"]
    try:
        diag_ids = list(pipeline.load([diag_inp])["input_ids"][0])
    except Exception:
        diag_ids = []

    for name in position_order:
        tp = positions_dict[name]
        try:
            idx = tp.index(diag_inp, is_original=True)
        except Exception as exc:
            idx = f"ERR: {type(exc).__name__}: {exc}"
        decoded = ""
        if isinstance(idx, list) and diag_ids:
            try:
                decoded = " | ".join(
                    repr(pipeline.tokenizer.decode([diag_ids[i]]))
                    for i in idx if 0 <= i < len(diag_ids)
                )
            except Exception:
                decoded = ""
        print(f"    {name:>14s} -> idx={idx}  tokens={decoded}")

    if resolver_errors:
        print("\nStructured-parser fallbacks detected:")
        for msg in resolver_errors:
            print("  -", msg)
        raise RuntimeError(
            "Structured parser fell back to `last_token` on at least one site. "
            "Fix the upstream cause before running patching."
        )

    if diag_ids:
        prefix_text = config.prompt_prefix or ""
        prefix_ids = pipeline.tokenizer(
            prefix_text, add_special_tokens=False
        )["input_ids"] if prefix_text else []
        prefix_len = len(prefix_ids)
        pad_id = pipeline.tokenizer.pad_token_id
        content_start = 0
        for i in range(len(diag_ids) - 1, -1, -1):
            if diag_ids[i] == pad_id:
                content_start = i + 1
                break
        prefix_span = (content_start, content_start + prefix_len)

        bad = []
        for name in position_order:
            if name == "last_token":
                continue
            try:
                ids = positions_dict[name].index(diag_inp, is_original=True)
            except Exception:
                continue
            if not isinstance(ids, list):
                continue
            for t in ids:
                if prefix_span[0] <= t < prefix_span[1]:
                    bad.append((name, t, pipeline.tokenizer.decode([diag_ids[t]])))
                    break
        if bad:
            print("\nPrefix-region intervention sites detected (would patch the prompt prefix, not the entities):")
            for name, t, tok in bad:
                print(f"  - {name} -> token {t} ({tok!r}) in prompt_prefix token span {prefix_span}")
            raise RuntimeError(
                "At least one entity site resolved to a token inside the prompt_prefix. "
                "Make sure `pipeline`, `config`, and the positional model were built "
                "in the order: config -> pipeline -> positional_model -> generators."
            )

    print("\nStructured parser resolved every site without falling back to last_token.")


__all__ = [
    'GRAPH_DEFAULT_COLOR',
    'GRAPH_TARGET_COLOR',
    'GRAPH_CONTROL_COLOR',
    'GRAPH_OUTPUT_COLOR',
    'show_localization_view',
    'set_seed',
    'make_dataset',
    'report_distinguishability',
    'proportion_true',
    'display_change_rate_table',
    'plot_change_rate_heatmap',
    'display_effect_rate_table',
    'plot_effect_rate_heatmap',
    'display_localization_scorecard',
    'plot_localization_scorecard',
    'plot_dataset_criteria_bars',
    'preview_counterfactual_pairs',
    'default_layers',
    'run_patching_for_datasets',
    'plot_iia_triplet',
    'plot_iia_target_control_random',
    # Shared counterfactual / printing helpers
    'print_counterfactual_pair_text',
    'print_one_pair_per_dataset',
    'print_confounding_and_nonconfounding',
    'build_counterfactual_dataset',
    'load_activation_patching_pipeline',
    'load_entity_binding_pipeline',
    'filter_and_report',
    # Entity-binding-specific helpers
    'install_entity_binding_shims',
    'install_entity_binding_legacy_adapter',
    'install_entity_binding_parser_patch',
    'make_entity_binding_generators',
    'build_entity_binding_token_positions',
    'diagnose_entity_binding_sites',
]
