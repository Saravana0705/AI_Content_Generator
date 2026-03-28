from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "image_benchmark_runs.csv"
PLOTS_DIR = BASE_DIR / "plots_fixed"
PLOTS_DIR.mkdir(exist_ok=True)

sns.set_theme(style="white", context="talk")
plt.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 300,
    "font.family": "DejaVu Sans",
    "axes.titleweight": "bold",
    "axes.grid": False,
})

MODEL_ORDER = [
    "OpenAI Image-1 Mini",
    "Stability SDXL",
    "Freepik Mystic",
]


def normalize_bool(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.upper()
        .map({
            "TRUE": True,
            "FALSE": False,
            "1": True,
            "0": False,
            "YES": True,
            "NO": False,
        })
        .fillna(False)
        .astype(bool)
    )


def get_model_label(row):
    provider = str(row.get("provider", "")).lower()
    model = str(row.get("model", "")).lower()

    if "openai" in provider:
        return "OpenAI Image-1 Mini"
    if "stability" in provider:
        return "Stability SDXL"
    if "freepik" in provider:
        return "Freepik Mystic"

    return model or provider or "Unknown"


def save_fig(fig, filename: str):
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def style_axis(ax):
    ax.grid(False)
    ax.xaxis.grid(False)
    ax.yaxis.grid(False)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.2)


def add_threshold_legend(ax):
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(0.995, 0.995),   # top-right corner inside axes
        fontsize=10,                     # smaller legend text
        handlelength=1.8,                # shorter dashed-line sample
        handletextpad=0.4,
        borderpad=0.25,
        labelspacing=0.2,
        borderaxespad=0.15,
        frameon=True,
        fancybox=True,
        framealpha=0.95,
    )


def draw_barplot(
    ax,
    summary_df,
    value_col,
    title,
    ylabel,
    colors,
    is_percent=False,
    ylim=None,
    threshold=None,
    threshold_fmt="{:.3f}",
    value_fmt="{:.2f}",
):
    x = np.arange(len(summary_df))
    y = summary_df[value_col].to_numpy(dtype=float)

    bars = ax.bar(
        x,
        y,
        color=colors,
        edgecolor="none",
        linewidth=0,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["model_label"], rotation=0, ha="center")
    ax.set_title(title, pad=16)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")

    if ylim is not None:
        ax.set_ylim(*ylim)

    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin if ymax > ymin else 1.0

    # Add more headroom so legend does not overlap bar labels
    ax.set_ylim(ymin, ymax + yrange * 0.16)

    if is_percent:
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    if threshold is not None and not pd.isna(threshold):
        ax.axhline(
            threshold,
            linestyle="--",
            linewidth=1.8,
            color="crimson",
            label=f"Threshold = {threshold_fmt.format(threshold)}",
        )
        add_threshold_legend(ax)

    ymin, ymax = ax.get_ylim()
    yrange = ymax - ymin if ymax > ymin else 1.0
    value_offset = yrange * 0.012

    for bar in bars:
        h = bar.get_height()
        xcenter = bar.get_x() + bar.get_width() / 2
        value_label = f"{h * 100:.1f}%" if is_percent else value_fmt.format(h)

        ax.text(
            xcenter,
            h + value_offset,
            value_label,
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
        )

    style_axis(ax)


def annotate_boxplot_medians(ax, data_df, x_col, y_col, order):
    medians = data_df.groupby(x_col, observed=True)[y_col].median().reindex(order)
    ymin, ymax = ax.get_ylim()
    yspan = ymax - ymin if ymax > ymin else 1.0

    for i, med in enumerate(medians):
        if pd.isna(med):
            continue
        ax.text(
            i,
            med + yspan * 0.02,
            f"{med:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                edgecolor="none",
                alpha=0.9,
            ),
            zorder=5,
        )


def final_axis_touches(ax):
    ax.tick_params(axis="x", labelrotation=0)
    ax.margins(x=0.08)


df = pd.read_csv(CSV_PATH)
df["model_label"] = df.apply(get_model_label, axis=1)
df["passed"] = normalize_bool(df["passed"]) if "passed" in df.columns else False
df["technical_ok"] = normalize_bool(df["technical_ok"]) if "technical_ok" in df.columns else False
df["scoring_ok"] = normalize_bool(df["scoring_ok"]) if "scoring_ok" in df.columns else False
df["retry_flag"] = pd.to_numeric(df.get("retries", 0), errors="coerce").fillna(0).astype(int) > 0

for col in [
    "clip_score",
    "aesthetic_score",
    "latency_sec",
    "retries",
    "clip_threshold",
    "aesthetic_threshold",
]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

present_models = [m for m in MODEL_ORDER if m in df["model_label"].unique()]
if not present_models:
    present_models = sorted(df["model_label"].dropna().unique().tolist())

df["model_label"] = pd.Categorical(df["model_label"], categories=present_models, ordered=True)

sort_cols = ["model_label"]
if "prompt_id" in df.columns:
    sort_cols.append("prompt_id")
df = df.sort_values(sort_cols).reset_index(drop=True)

summary = (
    df.groupby("model_label", observed=True)
    .agg(
        n=("model_label", "size"),
        pass_rate=("passed", "mean"),
        clip_mean=("clip_score", "mean"),
        aesthetic_mean=("aesthetic_score", "mean"),
        latency_mean=("latency_sec", "mean"),
        retry_rate=("retry_flag", "mean"),
    )
    .reset_index()
)

clip_threshold = (
    df["clip_threshold"].dropna().iloc[0]
    if "clip_threshold" in df.columns and df["clip_threshold"].dropna().size
    else None
)
aesthetic_threshold = (
    df["aesthetic_threshold"].dropna().iloc[0]
    if "aesthetic_threshold" in df.columns and df["aesthetic_threshold"].dropna().size
    else None
)

# 1 Pass rate
fig, ax = plt.subplots(figsize=(10, 6))
draw_barplot(
    ax=ax,
    summary_df=summary,
    value_col="pass_rate",
    title="Pass Rate by Model",
    ylabel="Pass Rate",
    colors=["#4C72B0", "#DD8452", "#55A868"],
    is_percent=True,
    ylim=(0, 1.02),
)
final_axis_touches(ax)
save_fig(fig, "01_pass_rate_by_model.png")

# 2 CLIP mean
fig, ax = plt.subplots(figsize=(10, 6))
draw_barplot(
    ax=ax,
    summary_df=summary,
    value_col="clip_mean",
    title="Mean CLIP Score by Model",
    ylabel="CLIP Score",
    colors=["#6BA292", "#4C8C94", "#3B6C91"],
    threshold=clip_threshold,
    value_fmt="{:.3f}",
)
final_axis_touches(ax)
save_fig(fig, "02_clip_score_mean.png")

# 3 Aesthetic mean
fig, ax = plt.subplots(figsize=(10, 6))
draw_barplot(
    ax=ax,
    summary_df=summary,
    value_col="aesthetic_mean",
    title="Mean Aesthetic Score by Model",
    ylabel="Aesthetic Score",
    colors=["#D17C6E", "#B45578", "#7F4477"],
    threshold=aesthetic_threshold,
    threshold_fmt="{:.3f}",
    value_fmt="{:.2f}",
)
final_axis_touches(ax)
save_fig(fig, "03_aesthetic_score_mean.png")

# 4 Latency mean
fig, ax = plt.subplots(figsize=(10, 6))
draw_barplot(
    ax=ax,
    summary_df=summary,
    value_col="latency_mean",
    title="Mean Latency by Model",
    ylabel="Latency (seconds)",
    colors=["#4F4A7A", "#4E81A0", "#5FB3A5"],
    value_fmt="{:.2f}",
)
final_axis_touches(ax)
save_fig(fig, "04_latency_mean.png")

# 5 Retry rate
fig, ax = plt.subplots(figsize=(10, 6))
draw_barplot(
    ax=ax,
    summary_df=summary,
    value_col="retry_rate",
    title="Retry Rate by Model",
    ylabel="Retry Rate",
    colors=["#A6A6A6", "#C02F62", "#DD8C6A"],
    is_percent=True,
    ylim=(0, 1.02),
)
final_axis_touches(ax)
save_fig(fig, "05_retry_rate.png")

# 6 CLIP distribution
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(
    data=df,
    x="model_label",
    y="clip_score",
    order=present_models,
    palette=["#7AB8A6", "#E59670", "#95A3C3"],
    width=0.45,
    whis=1.5,
    showfliers=False,
    linewidth=1.4,
    ax=ax,
)
if clip_threshold is not None and not pd.isna(clip_threshold):
    ax.axhline(
        clip_threshold,
        linestyle="--",
        linewidth=1.8,
        color="crimson",
        label=f"Threshold = {clip_threshold:.3f}",
    )
    add_threshold_legend(ax)

ax.set_title("Distribution of CLIP Scores by Model", pad=16)
ax.set_ylabel("CLIP Score")
ax.set_xlabel("")
ax.set_xticklabels(present_models, rotation=0, ha="center")

# Add headroom for legend and median labels
ymin, ymax = ax.get_ylim()
yrange = ymax - ymin if ymax > ymin else 1.0
ax.set_ylim(ymin, ymax + yrange * 0.12)

annotate_boxplot_medians(ax, df, "model_label", "clip_score", present_models)
style_axis(ax)
final_axis_touches(ax)
save_fig(fig, "06_clip_score_distribution.png")

# 7 Aesthetic distribution
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(
    data=df,
    x="model_label",
    y="aesthetic_score",
    order=present_models,
    palette=["#8FC0B6", "#E1E3AF", "#B9B6CF"],
    width=0.45,
    whis=1.5,
    showfliers=False,
    linewidth=1.4,
    ax=ax,
)
if aesthetic_threshold is not None and not pd.isna(aesthetic_threshold):
    ax.axhline(
        aesthetic_threshold,
        linestyle="--",
        linewidth=1.8,
        color="crimson",
        label=f"Threshold = {aesthetic_threshold:.3f}",
    )
    add_threshold_legend(ax)

ax.set_title("Distribution of Aesthetic Scores by Model", pad=16)
ax.set_ylabel("Aesthetic Score")
ax.set_xlabel("")
ax.set_xticklabels(present_models, rotation=0, ha="center")

# Add headroom for legend and median labels
ymin, ymax = ax.get_ylim()
yrange = ymax - ymin if ymax > ymin else 1.0
ax.set_ylim(ymin, ymax + yrange * 0.12)

annotate_boxplot_medians(ax, df, "model_label", "aesthetic_score", present_models)
style_axis(ax)
final_axis_touches(ax)
save_fig(fig, "07_aesthetic_score_distribution.png")

# 8 Prompt-level pass/fail heatmap
if "prompt_id" in df.columns:
    heat_df = (
        df.pivot_table(
            index="prompt_id",
            columns="model_label",
            values="passed",
            aggfunc="first",
        )
        .reindex(columns=present_models)
        .sort_index()
    )

    heat_values = heat_df.fillna(False).astype(int)
    annot = heat_df.replace({True: "Pass", False: "Fail"}).fillna("")

    fig_height = max(4.5, 0.55 * len(heat_values))
    fig, ax = plt.subplots(figsize=(10, fig_height))
    sns.heatmap(
        heat_values,
        annot=annot,
        fmt="",
        cmap=sns.color_palette(["#E41F26", "#2CA02C"], as_cmap=True),
        vmin=0,
        vmax=1,
        cbar=False,
        linewidths=0.6,
        linecolor="white",
        annot_kws={"fontsize": 11, "fontweight": "bold", "color": "white"},
        ax=ax,
    )
    ax.set_title("Prompt-level Pass/Fail Matrix", pad=16)
    ax.set_xlabel("")
    ax.set_ylabel("Prompt ID")
    final_axis_touches(ax)
    save_fig(fig, "08_prompt_pass_fail_heatmap.png")

summary_export = summary.copy()
summary_export["pass_rate"] = (summary_export["pass_rate"] * 100).round(2)
summary_export["retry_rate"] = (summary_export["retry_rate"] * 100).round(2)
summary_export = summary_export.round(4)
summary_export.to_csv(PLOTS_DIR / "benchmark_summary_table.csv", index=False)

print(f"Updated benchmark plots saved in: {PLOTS_DIR}")