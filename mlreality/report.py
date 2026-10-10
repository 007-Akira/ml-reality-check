
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def generate_report(results, output_dir="results"):
    """Save a benchmark table and paired-difference chart."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(results)

    # Preserve full-precision scores
    csv_path = output / "benchmark.csv"
    df.to_csv(csv_path, index=False)

    # Convert accuracy differences to percentage points
    differences_pp = df["difference"] * 100

    fig, ax = plt.subplots(figsize=(11, 5))

    colors = [
        "forestgreen" if value > 0
        else "royalblue" if value < 0
        else "gray"
        for value in differences_pp
    ]

    ax.bar(
        df["seed"].astype(str),
        differences_pp,
        color=colors
    )

    ax.axhline(0, color="black", linewidth=1)

    mean_difference = differences_pp.mean()

    ax.axhline(
        mean_difference,
        color="darkorange",
        linestyle="--",
        label=f"Mean difference: {mean_difference:+.2f} pp"
    )

    ax.set_title("Paired Model Accuracy Differences")
    ax.set_xlabel("Data-splitting seed")
    ax.set_ylabel("Random Forest - Logistic Regression (pp)")
    ax.legend()

    plt.xticks(rotation=45)
    plt.tight_layout()

    chart_path = output / "paired_differences.png"
    fig.savefig(chart_path, dpi=180)
    plt.close(fig)

    return csv_path, chart_path
