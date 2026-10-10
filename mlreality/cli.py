
import statistics
import typer
from mlreality.report import generate_report
from mlreality.statistics import analyze_results, interpret_analysis
from mlreality.provenance import create_run_directory, create_provenance, save_json

from mlreality.benchmark import run_benchmark

app = typer.Typer(
    help="ML Reality Check — reproducible ML benchmarking."
)


@app.callback()
def main():
    """ML Reality Check CLI."""
    pass


@app.command()
def run(
    seeds: int = typer.Option(
        5,
        min=2,
        help="Number of paired experimental runs."
    )
):
    """Compare two classification models."""

    typer.echo("\nML Reality Check")
    typer.echo("=" * 66)

    results, metadata = run_benchmark(seeds)
    output_dir, run_id = create_run_directory()
    save_json(output_dir / "provenance.json", create_provenance(metadata, run_id))
    csv_path, chart_path = generate_report(results, output_dir=output_dir)

    typer.echo(
        f"{'Seed':<8}"
        f"{'Logistic Reg.':>16}"
        f"{'Random Forest':>16}"
        f"{'Difference':>16}"
    )

    typer.echo("-" * 66)

    for row in results:
        typer.echo(
            f"{row['seed']:<8}"
            f"{row['logistic_regression']:>15.2%} "
            f"{row['random_forest']:>15.2%} "
            f"{row['difference'] * 100:>+14.2f} pp"
        )

    typer.echo("-" * 66)

    lr_scores = [
        r["logistic_regression"] for r in results
    ]

    rf_scores = [
        r["random_forest"] for r in results
    ]

    differences = [
        r["difference"] for r in results
    ]

    typer.echo(
        f"Mean Logistic Regression: "
        f"{statistics.mean(lr_scores):.2%}"
    )

    typer.echo(
        f"Mean Random Forest:       "
        f"{statistics.mean(rf_scores):.2%}"
    )

    mean_diff = statistics.mean(differences)

    typer.echo(
        f"Mean paired difference:   "
        f"{mean_diff * 100:+.2f} pp"
    )

    if len(differences) >= 2:
        typer.echo(
            f"Std. dev. of differences: "
            f"{statistics.stdev(differences) * 100:.2f} pp"
        )

    wins_rf = sum(d > 0 for d in differences)
    wins_lr = sum(d < 0 for d in differences)
    ties = sum(d == 0 for d in differences)

    typer.echo(
        f"Split wins: RF={wins_rf}, "
        f"LR={wins_lr}, Ties={ties}"
    )

    typer.echo(f"\nResults saved: {csv_path}")
    typer.echo(f"Chart saved:   {chart_path}\n")

    split = metadata["splits"][0]
    analysis = analyze_results(
        results, n_train=split["n_train"], n_test=split["n_test"]
    )
    interpretation = interpret_analysis(analysis, practical_threshold=0.01)
    save_json(output_dir / "analysis.json", {
        "analysis": analysis,
        "interpretation": interpretation,
        "test_type": "exploratory_corrected_resampled_t",
        "notes": [
            "Accuracy differences are Random Forest minus Logistic Regression.",
            "Inference uses an approximate dependence correction.",
            "A p-value is not the probability a model is better.",
            "Results are not independent external validation.",
        ],
    })

    typer.echo("\nStatistical Analysis")
    typer.echo("-" * 50)

    typer.echo(
        f"Corrected SE: "
        f"{analysis['corrected_se'] * 100:.3f} pp"
    )

    typer.echo(
        f"95% CI: "
        f"[{analysis['ci_lower'] * 100:+.3f}, "
        f"{analysis['ci_upper'] * 100:+.3f}] pp"
    )

    if analysis["p_value"] is not None:
        typer.echo(
            f"Corrected p-value: "
            f"{analysis['p_value']:.4f}"
        )
    else:
        typer.echo("Corrected p-value: unavailable")

    typer.echo(
        "Interpretation: exploratory, approximate "
        "statistical comparison."
    )
    typer.echo(f"Observed winner: {interpretation['observed_winner']}")
    typer.echo(f"Corrected interval direction: {interpretation['statistical_direction']}")
    typer.echo(f"Practical improvement supported: {interpretation['practical_improvement_supported']}")
    typer.echo(f"Run ID: {run_id}")
    typer.echo(f"Report directory: {output_dir}")
        


if __name__ == "__main__":
    app()
