"""缩小预算串行检查主实验worker与依赖链；不计为科研样本。"""

import argparse
import json
import os
import subprocess
import sys

import yaml
from a5000_pool import dependencies, tasks_for

from gpaco.config import config_hash
from gpaco.data import ROOT, write_json
from gpaco.hardware_inputs import safe_directory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    campaign = safe_directory(args.output)
    campaign.mkdir(parents=True, exist_ok=False)
    uuid = os.environ["CUDA_VISIBLE_DEVICES"]
    config = yaml.safe_load((ROOT / "configs/workloads/a5000_main_queue.yaml").read_text())
    config.update(scales=[100], cohort_generations=[1], paired_blocks=1, batch=2)
    config["search"].update(ants=4, iterations=3)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip()
    tasks = tasks_for(config)
    write_json(
        campaign / "campaign.json",
        {
            "config": config,
            "commit": commit,
            "config_hash": config_hash(config),
            "snapshot": str(ROOT),
            "smoke_only": True,
        },
    )
    for task in sorted(tasks, key=lambda t: (t["priority"], t["id"])):
        if dependencies(task, tasks, campaign) != "ready":
            raise RuntimeError("smoke 依赖没有按预期解锁")
        output = campaign / "tasks" / task["id"] / "attempt-001"
        output.mkdir(parents=True)
        job_path = output / "job.json"
        write_json(
            job_path,
            {
                "campaign_directory": str(campaign),
                "gpu_uuid": uuid,
                "config": config,
                "task": task,
                "snapshot": str(ROOT),
                "commit": commit,
            },
        )
        with (output / "worker.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/run_main_benchmark.py"),
                    "--job",
                    str(job_path),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        if not (output / "COMPLETE.json").exists():
            raise RuntimeError(f"smoke 没有完成：{output}")
        task["status"] = "completed"
        task["attempts"] = [{"job_path": str(job_path), "gpu_uuid": uuid, "host": "smoke"}]
        write_json(campaign / "queue.json", tasks)
    subprocess.run(
        [sys.executable, str(ROOT / "scripts/report_a5000_main.py"), "--campaign", str(campaign)],
        check=True,
    )
    write_json(
        campaign / "SMOKE_COMPLETE.json",
        {"passed": True, "scientific_sample": False, "tasks": len(tasks)},
    )
    print(json.dumps({"smoke_passed": True, "output": str(campaign)}))


if __name__ == "__main__":
    main()
