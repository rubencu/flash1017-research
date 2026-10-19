"""Plot the completed fixed comparison; never render a partial run as accepted."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def gm(xs):
    return math.exp(statistics.mean(math.log(x) for x in xs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    supervisor = json.loads((args.results / "supervisor.json").read_text())
    if supervisor["status"] != "complete" or supervisor["restoration_status"] != "restored":
        raise RuntimeError("Require completed comparison and restored app")
    records = []
    sources = {}
    for process in supervisor["processes"]:
        if process["exit_code"] != 0:
            raise RuntimeError("A benchmark process failed")
        path = Path(process["output"])
        data = path.read_bytes()
        sources[path.name] = hashlib.sha256(data).hexdigest()
        rows = [json.loads(line) for line in data.splitlines()]
        for row in rows:
            if row["type"] == "run" and not row["warmup"]:
                if row["generated_tokens"] != 512 or not row["fixed_length_valid"] or not row["candidate"]["passed"]:
                    raise RuntimeError("Incomplete/invalid fixed-length measurement; do not plot accepted results")
                records.append(row)
    if len(records) != 48:
        raise RuntimeError("Expected all 48 measured requests")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    arms = ["native2048", "native8192", "complete"]
    labels = ["Stock · 2K chunks", "Stock · 8K chunks", "Complete runtime"]
    colors = ["#939DA8", "#487997", "#167B66"]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 5.2), sharey=True)
    chart = {"scope": "Geometric mean prompt+512-output-token time; whiskers show two process-block means, not confidence intervals",
             "source_hashes": sources, "groups": []}
    for ax, length, name in zip(axes, [1024, 16384, 65536], ["1K · synthetic code/prose", "16K · four natural sources", "64K · two natural sources"]):
        groups = [[r for r in records if r["input_tokens"] == length and r["arm"] == a] for a in arms]
        values = [gm([r["total_seconds"] for r in group]) for group in groups]
        block_values = [[gm([r["total_seconds"] for r in group if r["block"] == block]) for block in (0, 1)] for group in groups]
        low = [v-min(bs) for v, bs in zip(values, block_values)]
        high = [max(bs)-v for v, bs in zip(values, block_values)]
        ax.barh(range(3), values, color=colors, height=.55, xerr=[low, high],
                error_kw={"ecolor": "#25313A", "capsize": 3, "elinewidth": 1})
        for y, value in enumerate(values):
            ax.text(max(value, max(block_values[y])) + max(values)*.035, y, f"{value:.1f}s", va="center", fontsize=11)
        ax.set_title(name, fontsize=11, pad=15)
        ax.set_xlabel("Seconds · lower is better")
        ax.set_xlim(0, max(max(bs) for bs in block_values)*1.28)
        ax.grid(axis="x", alpha=.15)
        ax.set_axisbelow(True)
        ax.set_yticks(range(3), labels)
        chart["groups"].append({"input_tokens": length, "arms": [
            {"arm": a, "runs": len(g), "geomean_seconds": v, "block_geomeans": b}
            for a, g, v, b in zip(arms, groups, values, block_values)]})
    axes[0].invert_yaxis()
    fig.suptitle("All Flash1017 changes together", x=.05, ha="left", fontsize=20, fontweight="bold", y=.98)
    fig.text(.05, .89, "Complete runtime versus untouched installed oMLX 0.7.1rc1 with native speculative decoding", fontsize=11)
    fig.text(.05, .09, "M5 Ultra · 80 GPU cores · 256 GiB · exact oQ2.5e-mtp checkpoint · 2 fresh processes per configuration", fontsize=9)
    fig.text(.05, .045, "Warm-model timing; excludes loading and HTTP. Fresh request KV; retained row/OS cache history is documented. Output streams may differ.", fontsize=8.5)
    fig.subplots_adjust(left=.17, right=.98, bottom=.24, top=.77, wspace=.30)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        path = args.output.with_suffix(suffix)
        if path.exists():
            raise FileExistsError(path)
        fig.savefig(path, dpi=180, facecolor="white")
    args.output.with_suffix(".json").write_text(json.dumps(chart, indent=2) + "\n")
    plt.close(fig)


if __name__ == "__main__":
    main()
