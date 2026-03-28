import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

LOG_FILE = "runs/runs_log.csv"
OUT_DIR = "runs/plots"

# Cloud comparison set
CLOUD_MODELS = [
    "gpt-4o",
    "llama-3.3-70b-versatile",
    "moonshotai/kimi-k2-instruct",
    "qwen/qwen3-32b",
]

NUM_COLS = [
    "optimized_score",
    "total_time_sec",
    "speed_wps",
    "score_improvement",
    "revision_rounds",
    "initial_score",
    "revised_score",
]

# Default date filter (UTC date in the CSV timestamp)
# Override in PowerShell:
#   $env:RUN_DATE="2026-01-30"
RUN_DATE = os.getenv("RUN_DATE", "2026-01-30")

# Label offsets to reduce overlap in tradeoff/Pareto charts
LABEL_OFFSETS = {
    "gpt-4o": (8, 8),
    "llama-3.3-70b-versatile": (8, -12),
    "moonshotai/kimi-k2-instruct": (8, 8),
    "qwen/qwen3-32b": (8, 8),
}


def scatter(df, x, y, title, xlabel, ylabel, fname, logx=False):
    """Per-run scatter plot grouped by model."""
    plt.figure()
    for model, g in df.groupby("model_name"):
        plt.scatter(
            g[x],
            g[y],
            label=model,
            alpha=0.7,
            s=35,
        )
    if logx:
        plt.xscale("log")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname))
    plt.close()


def bar(df, x, y, title, ylabel, fname, fmt="{:.1f}"):
    """Bar plot with value labels on each bar."""
    plt.figure()
    bars = plt.bar(df[x], df[y])
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=30, ha="right")

    # Add value labels on bars
    for b in bars:
        val = b.get_height()
        if np.isnan(val):
            continue
        plt.text(
            b.get_x() + b.get_width() / 2,
            val,
            fmt.format(val),
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, fname))
    plt.close()


def pareto_frontier(df_points):
    """
    df_points must have columns: avg_time_sec, avg_score
    Frontier: minimize time, maximize score
    """
    pareto = df_points.sort_values("avg_time_sec").copy()
    frontier = []
    best_score = -np.inf
    for _, r in pareto.iterrows():
        if r["avg_score"] > best_score:
            frontier.append(r)
            best_score = r["avg_score"]
    return pd.DataFrame(frontier)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = pd.read_csv(LOG_FILE)

    # Parse timestamp (CSV is ISO with timezone)
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["date_utc"] = df["timestamp_dt"].dt.strftime("%Y-%m-%d")

    # Filter to today's runs
    df = df[df["date_utc"] == RUN_DATE].copy()

    # Filter to cloud models only
    df = df[df["model_name"].isin(CLOUD_MODELS)].copy()

    # Convert numeric columns safely
    for col in NUM_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows missing key metrics
    df = df.dropna(subset=["prompt_id", "model_name", "optimized_score", "total_time_sec"])

    # Keep latest run per (model, prompt_id)
    df = df.sort_values("timestamp_dt").drop_duplicates(subset=["model_name", "prompt_id"], keep="last")

    # ---- Coverage check: ensure equal prompts per model ----
    counts = df.groupby("model_name")["prompt_id"].nunique().reindex(CLOUD_MODELS)
    common_prompts = None
    for m in CLOUD_MODELS:
        s = set(df[df["model_name"] == m]["prompt_id"].unique())
        common_prompts = s if common_prompts is None else (common_prompts & s)

    common_n = len(common_prompts) if common_prompts is not None else 0

    print(f"\n=== Plotting Benchmark Results for date (UTC): {RUN_DATE} ===")
    print("Unique prompts per model:")
    for m in CLOUD_MODELS:
        print(f"  {m:28s}: {int(counts.loc[m]) if pd.notna(counts.loc[m]) else 0}")
    print(f"Common prompts across all 4 models: {common_n}")

    # Force fairness: use common prompts only
    df = df[df["prompt_id"].isin(common_prompts)].copy()

    # Aggregate summary per model (on common prompts)
    agg = df.groupby("model_name", as_index=False).agg(
        avg_time_sec=("total_time_sec", "mean"),
        avg_speed_wps=("speed_wps", "mean"),
        avg_score=("optimized_score", "mean"),
        rev_rate=("revision_rounds", lambda s: (s > 0).mean() * 100.0),
    )

    # Baseline = worst model by mean optimized score (for THIS run-date + prompt set)
    baseline_row = agg.loc[agg["avg_score"].idxmin()]
    BASELINE_MODEL = baseline_row["model_name"]
    baseline_score = float(baseline_row["avg_score"])
    baseline_time = float(baseline_row["avg_time_sec"])

    print(f"\nBenchmark (worst) model on {common_n} shared prompts: {BASELINE_MODEL}")
    print(f"  baseline avg_score = {baseline_score:.2f}")
    print(f"  baseline avg_time  = {baseline_time:.2f} sec\n")

    # ---------------------------
    # PLOTS (Cloud, Today only)
    # ---------------------------

    # A1: Average optimized score
    bar(
        agg.sort_values("avg_score"),
        "model_name",
        "avg_score",
        f"Average Optimized Score (Cloud Models) — {RUN_DATE} (n={common_n})",
        "Optimized Score",
        "A1_avg_score_cloud_today.png",
        fmt="{:.1f}",
    )

    # A2: Time vs optimized score (per-run scatter)
    scatter(
        df,
        "total_time_sec",
        "optimized_score",
        f"Time vs Optimized Score (Cloud Models) — {RUN_DATE} (n={common_n})",
        "Total Time (sec)",
        "Optimized Score",
        "A2_time_vs_score_cloud_today.png",
    )

    # A3: Speed vs optimized score (per-run scatter)
    scatter(
        df,
        "speed_wps",
        "optimized_score",
        f"Speed vs Optimized Score (Cloud Models) — {RUN_DATE} (n={common_n})",
        "Speed (words/sec)",
        "Optimized Score",
        "A3_speed_vs_score_cloud_today.png",
    )

    # A4: Avg improvement (revised runs only)
    revised = df[(df["revision_rounds"] > 0) & (df["score_improvement"] > 0)].copy()
    if not revised.empty:
        avg_improve = revised.groupby("model_name", as_index=False)["score_improvement"].mean()
        bar(
            avg_improve.sort_values("score_improvement"),
            "model_name",
            "score_improvement",
            f"Average Score Improvement (Revised Runs) — {RUN_DATE}",
            "Score Improvement",
            "A4_avg_improvement_cloud_today.png",
            fmt="{:.1f}",
        )

    # A5: Revision rate % (show integer labels)
    bar(
        agg.sort_values("rev_rate"),
        "model_name",
        "rev_rate",
        f"Revision Rate (Cloud Models) — {RUN_DATE} (n={common_n})",
        "Revision Rate (%)",
        "A5_revision_rate_cloud_today.png",
        fmt="{:.0f}",
    )

    # E1: Tradeoff plot (avg time vs avg score), baseline highlighted + ref lines
    plt.figure()
    for _, r in agg.iterrows():
        name = r["model_name"]
        x = float(r["avg_time_sec"])
        y = float(r["avg_score"])

        if name == BASELINE_MODEL:
            plt.scatter(x, y, s=240, marker="X", label=f"{name} (baseline)")
        else:
            plt.scatter(x, y, s=140, label=name)

        dx, dy = LABEL_OFFSETS.get(name, (6, 6))
        plt.annotate(name, (x, y), textcoords="offset points", xytext=(dx, dy), ha="left")

    plt.axhline(baseline_score, linestyle="--")
    plt.axvline(baseline_time, linestyle="--")
    plt.xlabel("Average Total Time per Prompt (sec)")
    plt.ylabel("Average Optimized Score")
    plt.title(f"Quality–Latency Tradeoff (Cloud Models) — {RUN_DATE} (Baseline auto={BASELINE_MODEL})")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "E1_tradeoff_time_vs_score_cloud_today.png"))
    plt.close()

    # E2: Pareto frontier (labeled)
    frontier = pareto_frontier(agg[["model_name", "avg_time_sec", "avg_score"]].copy())

    plt.figure()
    for _, r in agg.iterrows():
        name = r["model_name"]
        x = float(r["avg_time_sec"])
        y = float(r["avg_score"])
        if name == BASELINE_MODEL:
            plt.scatter(x, y, s=240, marker="X", label=f"{name} (baseline)")
        else:
            plt.scatter(x, y, s=140, label=name)

        dx, dy = LABEL_OFFSETS.get(name, (6, 6))
        plt.annotate(name, (x, y), textcoords="offset points", xytext=(dx, dy), ha="left")

    plt.plot(frontier["avg_time_sec"], frontier["avg_score"], linestyle="--", linewidth=2, label="Pareto frontier")

    plt.xlabel("Average Total Time per Prompt (sec)")
    plt.ylabel("Average Optimized Score")
    plt.title(f"Pareto Frontier: Time vs Optimized Score (Cloud Models) — {RUN_DATE}")
    plt.legend()
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "E2_pareto_frontier_cloud_today.png"))
    plt.close()

    print("Success: Updated plots generated in runs/plots/")


if __name__ == "__main__":
    main()
