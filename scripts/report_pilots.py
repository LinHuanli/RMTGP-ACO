"""汇总已提交的 history，绘制训练/验证 gap 和代耗时；不读取测试集。"""

import argparse
import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gpaco.data import ROOT, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", default="pilot-v1")
    args = parser.parse_args()
    directory = (ROOT / "artifacts" / args.campaign).resolve()
    if not directory.is_relative_to(ROOT / "artifacts"):
        raise ValueError("只汇总项目内实验")
    rows = []
    for path in sorted(directory.glob("*/history.json")):
        run = path.parent
        history = json.loads(path.read_text())
        if not history:
            continue
        config = json.loads((run / "run_manifest.json").read_text())
        complete = (run / "COMPLETE.json").exists()
        x = [r["generation"] for r in history]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
        for key, label in [
            ("train_best_gap_percent", "Generation best"),
            ("train_median_gap_percent", "Population median"),
            ("train_baseline_gap_percent", "Fixed ACO"),
        ]:
            axes[0].plot(x, [r[key] for r in history], label=label)
        validated = [r for r in history if r.get("validation_champion_gap_percent") is not None]
        for key, label in [
            ("validation_champion_gap_percent", "Selected champion"),
            ("validation_baseline_gap_percent", "Fixed ACO"),
        ]:
            axes[1].plot(
                [r["generation"] for r in validated],
                [r.get(key) for r in validated],
                marker="o",
                label=label,
            )
        for key, label in [
            ("generation_wall_s", "Generation wall"),
            ("eval_wall_s", "Evaluation"),
            ("validation_s", "Validation incl. reference setup"),
        ]:
            axes[2].plot(x, [r.get(key, 0) for r in history], label=label)
        for axis, title in zip(
            axes, ["Training batches", "Fixed validation", "Time per generation"], strict=True
        ):
            axis.set(xlabel="GP generation", title=title)
            axis.legend(fontsize=8)
            axis.grid(alpha=0.2)
        axes[0].set_ylabel("Gap to supplied label (%)")
        axes[1].set_ylabel("Gap to supplied label (%)")
        axes[2].set_ylabel("Seconds")
        fig.suptitle(f"{run.name} ({'complete' if complete else 'partial'})")
        fig.savefig(run / "curves.png", dpi=160)
        fig.savefig(run / "curves.pdf")
        plt.close(fig)
        row = {
            "run": run.name,
            "n": config["n"],
            "seed": config["root_seed"],
            "completed": complete,
            "generations": len(history),
            "latest_train_gap": history[-1]["train_best_gap_percent"],
            "median_evaluation_s": sorted(r["eval_wall_s"] for r in history)[len(history) // 2],
            "validation_gap": validated[-1]["validation_champion_gap_percent"]
            if validated
            else None,
            "validation_delta_pp": validated[-1].get("validation_delta_pp") if validated else None,
        }
        rows.append(row)
    target = directory / "summary"
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / "runs.json", rows)
    if rows:
        with (target / "runs.csv").open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
