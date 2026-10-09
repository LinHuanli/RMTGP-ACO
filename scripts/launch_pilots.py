"""先列出任务；显式 --execute 才将 6 个独立 pilot 放到空闲 A5000。"""

import argparse
import csv
import io
import json
import shlex
import subprocess
import tarfile
from concurrent.futures import ThreadPoolExecutor

from gpaco.data import ROOT, write_json

SERVERS = ["cuda08", "cuda01", "cuda02", "cuda03", "cuda11", "cuda04", "cuda05", "cuda06"]


def remote(host, command):
    return subprocess.check_output(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host, command],
        text=True,
        timeout=45,
    )


def idle_devices(host):
    try:
        rows = remote(
            host,
            "nvidia-smi --query-gpu=uuid,name,utilization.gpu,memory.used "
            "--format=csv,noheader,nounits",
        )
        apps = remote(
            host, "nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits"
        )
        used = {row[0].strip() for row in csv.reader(apps.splitlines()) if len(row) >= 2}
        return [
            {"host": host, "gpu_uuid": row[0].strip(), "model": row[1].strip()}
            for row in csv.reader(rows.splitlines())
            if len(row) == 4
            and row[1].strip() == "NVIDIA RTX A5000"
            and row[0].strip() not in used
            and int(row[2]) <= 5
            and int(row[3]) <= 1024
        ]
    except (subprocess.SubprocessError, ValueError) as error:
        print(f"跳过 {host}: {error}")
        return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--campaign", default="pilot-v1")
    parser.add_argument("--limit", type=int, default=6)
    args = parser.parse_args()
    if not args.campaign.replace("-", "").replace("_", "").isalnum():
        raise ValueError("campaign 只能包含字母、数字、连字符和下划线")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if args.execute:
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True
        )
        if dirty:
            raise RuntimeError("先提交已跟踪文件，再用不可变快照启动")
        untracked = subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard", "src", "scripts", "configs"],
            cwd=ROOT,
            text=True,
        )
        if untracked:
            raise RuntimeError("源码/脚本/配置含有未提交新文件，不能将其漏出运行快照")
    with ThreadPoolExecutor(max_workers=8) as pool:
        devices = [device for group in pool.map(idle_devices, SERVERS) for device in group]
    tasks = [(n, seed) for n in (100, 500) for seed in (1001, 1002, 1003)]
    pending = []
    for n, seed in tasks:
        name = f"as-tsp{n}-seed{seed}"
        launch = ROOT / "artifacts/launches" / args.campaign / name
        if not (launch / "job.json").exists():
            pending.append((n, seed, name, launch))
    allocations = list(zip(pending[: args.limit], devices, strict=False))
    print(
        json.dumps(
            {
                "commit": commit,
                "available": devices,
                "to_launch": [{"n": task[0], "seed": task[1], **gpu} for task, gpu in allocations],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not args.execute or not allocations:
        return
    snapshot = ROOT / "artifacts/runtime" / commit
    if not snapshot.exists():
        archive = subprocess.check_output(
            ["git", "archive", commit, "src", "scripts", "configs", "pyproject.toml"], cwd=ROOT
        )
        snapshot.mkdir(parents=True)
        with tarfile.open(fileobj=io.BytesIO(archive)) as package:
            package.extractall(snapshot, filter="data")
    for (n, seed, name, launch), gpu in allocations:
        output = ROOT / "artifacts" / args.campaign / name
        command = ["train", "--n", str(n), "--seed", str(seed), "--output", str(output)]
        job = {**gpu, "snapshot": str(snapshot), "commit": commit, "commands": [command]}
        launch.mkdir(parents=True)
        job_path = launch / "job.json"
        write_json(job_path, job)
        script = (
            f"cd {shlex.quote(str(ROOT))}\nsource scripts/env.sh\n"
            f"nohup setsid python {shlex.quote(str(snapshot / 'scripts/run_worker.py'))} "
            f"--job {shlex.quote(str(job_path))} "
            f"> {shlex.quote(str(launch / 'worker.log'))} 2>&1 < /dev/null &\n"
            'GPACO_WORKER_PID=$!\nsleep 1\nps -p "$GPACO_WORKER_PID" -o pid=,args=\n'
        )
        try:
            result = remote(gpu["host"], "bash -c " + shlex.quote(script))
            write_json(launch / "launched.json", {"process": result.strip(), **gpu})
            print(f"已启动 {name}: {gpu['host']} {result.strip()}", flush=True)
        except subprocess.SubprocessError as error:
            write_json(launch / "launch_failure.json", {"error": str(error)})
            print(f"启动失败 {name}，保留记录，未自动重试：{error}", flush=True)


if __name__ == "__main__":
    main()
