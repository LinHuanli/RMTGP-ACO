"""完整诊断文件的流式 SHA256 核验；只在独立报告目录写派生缓存。"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import time

from .common import atomic_json, digest, file_hash, now, read_json
from .report_inputs import diagnostic_coverage


def inventory(out):
    """只追踪冻结队列及其显式子任务，不扫描归档或失败重跑目录。"""
    out = Path(out)
    jobs = {}
    for path in sorted((out / "queue").glob("*.json")):
        task = read_json(path)["task"]
        parent = out / "jobs" / task["id"]
        status = read_json(parent / "status.json")
        if status.get("status") != "completed":
            raise ValueError(f"任务未完成: {task['id']}")
        if task["kind"] == "numeric_pair":
            comparison = read_json(parent / "comparison.json")
            if comparison["task"] != task:
                raise ValueError("数值对照身份与队列不一致")
            for mode, child in comparison["children"].items():
                if child["id"] != task["id"] + "--" + mode:
                    raise ValueError("数值子任务身份错误")
                jobs[child["id"]] = out / "jobs" / child["id"] / "diagnostics"
        elif task["kind"] == "historical_mechanism":
            jobs[task["id"]] = parent / "diagnostics"
        elif task["kind"] == "numeric_validation":
            # 验收内部也保存 schema 3，但只有 100 轮，不能进入质量统计。
            for idx in sorted(parent.glob("*/diagnostics/index.json")):
                jobs[task["id"] + "/" + idx.parent.parent.name] = idx.parent
    return jobs


def audit_one(args):
    name, directory, cache_dir, code_hash = args
    directory, cache_dir = Path(directory), Path(cache_dir)
    started = time.monotonic()
    index = read_json(directory / "index.json")
    flats = diagnostic_coverage(index)
    stamps = {}
    for filename, record in index["files"].items():
        path = directory / filename
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("诊断索引路径越界")
        stat = path.stat()
        if stat.st_size != record["compressed_bytes"]:
            raise ValueError(f"诊断文件大小变化: {path}")
        stamps[filename] = [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]
    key = digest({"index": file_hash(directory / "index.json"), "stamps": stamps, "analysis": code_hash})
    cache = cache_dir / (digest(name) + ".json")
    prior = read_json(cache, {})
    if prior.get("cache_key") == key:
        return {**prior, "cache_reused": True}
    for filename, record in index["files"].items():
        if file_hash(directory / filename) != record["sha256"]:
            raise ValueError(f"诊断文件 SHA256 不一致: {directory / filename}")
        stat = (directory / filename).stat()
        if stamps[filename] != [stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]:
            raise ValueError("校验期间输入发生变化")
    result = {"job": name, "directory": str(directory), "status": "passed", "cache_key": key,
              "index_sha256": file_hash(directory / "index.json"), "files": len(index["files"]),
              "bytes": sum(s[0] for s in stamps.values()), "horizon": index["horizon"],
              "executed_solves": len(flats), "audited_at": now(), "seconds": time.monotonic()-started,
              "cache_reused": False}
    atomic_json(cache, result)
    return result


def run(out, report_dir, workers=4):
    report_dir = Path(report_dir)
    jobs = inventory(out)
    code = digest({p.name:file_hash(p) for p in (Path(__file__), Path(__file__).with_name("report_inputs.py"))})
    cache = report_dir / ".cache" / "audit"
    results, errors = [], []
    atomic_json(report_dir / "audit_progress.json", {"status":"running", "total":len(jobs), "done":0})
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(audit_one, (name, path, cache, code)):name for name,path in jobs.items()}
        for future in as_completed(pending):
            try:
                result = future.result()
                results.append(result)
                print(f"audit {len(results)+len(errors)}/{len(jobs)} {result['job']} {result['bytes']/1e9:.2f} GB", flush=True)
            except Exception as error:
                errors.append({"job":pending[future], "error":repr(error)})
            atomic_json(report_dir / "audit_progress.json", {"status":"running", "total":len(jobs),
                        "done":len(results)+len(errors), "errors":errors})
    summary = {"status":"failed" if errors else "passed", "checked_at":now(), "jobs":len(jobs),
               "files":sum(r["files"] for r in results), "bytes":sum(r["bytes"] for r in results),
               "results":sorted(results,key=lambda r:r["job"]), "errors":errors,
               "scope":"索引覆盖与全部引用文件 SHA256；不等于重新执行 terminal 数学 oracle",
               "cache_policy":"内容哈希首次全读；仅输入大小、mtime、ctime、索引与分析代码摘要均不变时复用"}
    atomic_json(report_dir / "integrity_audit.json", summary)
    atomic_json(report_dir / "audit_progress.json", {"status":summary["status"],"total":len(jobs),"done":len(jobs),"errors":errors})
    if errors:
        raise ValueError(f"{len(errors)} 个诊断任务未通过校验")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    run(args.output, args.report_dir, args.workers)
