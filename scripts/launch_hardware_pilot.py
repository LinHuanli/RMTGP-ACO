"""显式启动五卡异构先导；固定源码快照、UUID、项目内路径与 nohup 日志。"""

import argparse
import csv
import io
import json
import shlex
import subprocess
import tarfile
from concurrent.futures import ThreadPoolExecutor

import yaml
from launch_pilots import remote

from gpaco.config import config_hash
from gpaco.data import ROOT, write_json
from gpaco.hardware_inputs import safe_directory


def check_target(item):
    model, target = item
    device = remote(
        target["host"],
        (
            f"nvidia-smi -i {shlex.quote(target['gpu_uuid'])} "
            "--query-gpu=uuid,name,utilization.gpu,memory.used --format=csv,noheader,nounits"
        ),
    )
    row = next(csv.reader(device.splitlines()))
    apps = remote(
        target["host"], "nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits"
    )
    pids = [
        r[1].strip()
        for r in csv.reader(apps.splitlines())
        if len(r) >= 2 and r[0].strip() == target["gpu_uuid"]
    ]
    idle = not pids and int(row[2]) <= 5 and int(row[3]) <= 1024
    if row[1].strip() != target["model"]:
        raise RuntimeError(f"{model} GPU 型号变化：{row}")
    return {"key": model, **target, "idle": idle, "pids": pids, "state": row}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hardware/cross_gpu_pilot.yaml")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    config_path = safe_directory(ROOT / args.config)
    config = yaml.safe_load(config_path.read_text())
    campaign = safe_directory(ROOT / "artifacts" / config["name"])
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if args.execute:
        changed = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=ROOT,
            text=True,
        )
        untracked = subprocess.check_output(
            [
                "git",
                "ls-files",
                "--others",
                "--exclude-standard",
                "src",
                "scripts",
                "configs",
                "tests",
            ],
            cwd=ROOT,
            text=True,
        )
        if changed or untracked:
            raise RuntimeError("运行源码/配置/测试需先提交；不使用可变工作树启动")
        if campaign.exists():
            raise FileExistsError("实验目录已存在，禁止覆盖或重复启动；请显式规划恢复")
    with ThreadPoolExecutor(max_workers=5) as pool:
        targets = list(pool.map(check_target, config["targets"].items()))
    print(
        json.dumps(
            {"commit": commit, "config_hash": config_hash(config), "targets": targets}, indent=2
        )
    )
    if not args.execute:
        return
    if not all(t["idle"] for t in targets):
        raise RuntimeError("有目标卡已被占用，尚未启动任何任务")
    snapshot = ROOT / "artifacts/runtime" / commit
    if not snapshot.exists():
        archive = subprocess.check_output(
            ["git", "archive", commit, "src", "scripts", "configs", "tests", "pyproject.toml"],
            cwd=ROOT,
        )
        snapshot.mkdir(parents=True)
        with tarfile.open(fileobj=io.BytesIO(archive)) as package:
            package.extractall(snapshot, filter="data")
    if not (snapshot / "tests/test_cuda.py").exists():
        raise RuntimeError("快照缺少 E00 测试，拒绝启动")
    campaign.mkdir(parents=True)
    write_json(
        campaign / "campaign.json",
        {
            "config": config,
            "config_hash": config_hash(config),
            "commit": commit,
            "snapshot": str(snapshot),
            "standard_test_opened": False,
        },
    )
    for target in targets:
        model = target["key"]
        directory = campaign / "devices" / model
        directory.mkdir(parents=True)
        job = {
            "config": config,
            "model": model,
            "snapshot": str(snapshot),
            "commit": commit,
            "campaign_directory": str(campaign),
        }
        job_path = directory / "job.json"
        write_json(job_path, job)
        script = (
            f"cd {shlex.quote(str(ROOT))}\nsource scripts/env.sh\n"
            f"export CUDA_VISIBLE_DEVICES={shlex.quote(target['gpu_uuid'])}\n"
            f"export GPACO_COMMIT={shlex.quote(commit)} GPACO_SNAPSHOT={shlex.quote(str(snapshot))}\n"
            f"export PYTHONPATH={shlex.quote(str(snapshot / 'src'))}\n"
            f"nohup setsid python -m gpaco.hardware_campaign --job {shlex.quote(str(job_path))} "
            f"> {shlex.quote(str(directory / 'worker.log'))} 2>&1 < /dev/null &\n"
            'GPACO_HW_PID=$!\nsleep 1\nps -p "$GPACO_HW_PID" -o pid=,args=\n'
        )
        try:
            process = remote(target["host"], "bash -c " + shlex.quote(script))
            write_json(
                directory / "launched.json", {"process": process.strip(), "host": target["host"]}
            )
            print(f"已启动 {model}: {target['host']} {process.strip()}", flush=True)
        except subprocess.SubprocessError as error:
            # 不静默重试，也不取消已经开始的独立实验。
            write_json(directory / "FAILED.json", {"stage": "launch", "error": str(error)})
            print(f"{model} 启动失败：{error}", flush=True)


if __name__ == "__main__":
    main()
