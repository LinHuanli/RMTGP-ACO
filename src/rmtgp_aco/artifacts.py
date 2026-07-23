"""运行目录、Git 状态、数据快照和 seed provenance。"""

from __future__ import annotations

from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

from .config import ExperimentConfig
from .runtime import runtime_state


def _project_version() -> str:
    try:
        return version("rmtgp-aco")
    except PackageNotFoundError:
        return "0.2.0+uninstalled"


def _git_output(repository: Path, *arguments: str) -> str | None:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def git_state(repository: str | Path) -> dict[str, Any]:
    """读取 commit、branch 与 dirty 状态；尚无首个 commit 时显式记录。"""

    root = Path(repository).resolve()
    commit = _git_output(root, "rev-parse", "HEAD")
    branch = _git_output(root, "branch", "--show-current")
    status = _git_output(root, "status", "--porcelain")
    return {
        "commit_sha": commit,
        "branch": branch,
        "dirty": bool(status),
        "status_entries": len(status.splitlines()) if status else 0,
    }


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def initialise_run_artifacts(
    output_directory: str | Path,
    experiment: ExperimentConfig,
    *,
    repository: str | Path,
    data_manifest: str | Path | None = None,
) -> dict[str, Any]:
    """建立 run manifest、seed 说明，并复制数据 manifest 快照。"""

    target = Path(output_directory)
    target.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(UTC).isoformat()
    data_payload: dict[str, Any] | None = None
    if data_manifest is not None:
        source = Path(data_manifest)
        copied = target / "data_manifest.json"
        shutil.copyfile(source, copied)
        data_payload = {
            "source": source.resolve().as_posix(),
            "snapshot": copied.name,
            "sha256": _file_sha256(copied),
        }

    seeds = {
        "root_seed": experiment.root_seed,
        "derivation": (
            "NumPy SeedSequence([root_seed, generation/scale/batch/replicate, "
            "domain-separation constant])"
        ),
        "common_random_numbers": True,
    }
    (target / "seeds.json").write_text(
        json.dumps(seeds, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    payload = {
        "schema_version": 1,
        "project_version": _project_version(),
        "experiment_id": experiment.experiment_id,
        "aco_config_hash": experiment.aco.config_hash,
        "configuration": experiment.stable_dict(),
        "git": git_state(repository),
        "runtime": runtime_state(experiment.runtime),
        "data_manifest": data_payload,
        "started_at": started_at,
        "ended_at": None,
        "status": "running",
        "error": None,
    }
    _write_run_manifest(target, payload)
    return payload


def finalise_run_artifacts(
    output_directory: str | Path,
    payload: dict[str, Any],
    *,
    status: str,
    error: str | None = None,
) -> None:
    """将 run 状态原子地收束为 completed 或 failed。"""

    if status not in {"completed", "failed"}:
        raise ValueError("status 必须为 completed 或 failed")
    payload = dict(payload)
    payload["ended_at"] = datetime.now(UTC).isoformat()
    payload["status"] = status
    payload["error"] = error
    _write_run_manifest(Path(output_directory), payload)


def resume_run_artifacts(
    output_directory: str | Path,
) -> dict[str, Any]:
    """把本地未完成 run 重新标记为 running，并保留原始 provenance。"""

    target = Path(output_directory)
    source = target / "manifest.json"
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("run manifest 根对象必须为 mapping")
    payload = dict(payload)
    payload["status"] = "running"
    payload["ended_at"] = None
    payload["error"] = None
    payload["resume_count"] = int(payload.get("resume_count", 0)) + 1
    payload["resumed_at"] = datetime.now(UTC).isoformat()
    _write_run_manifest(target, payload)
    return payload


def _write_run_manifest(target: Path, payload: dict[str, Any]) -> None:
    temporary = target / "manifest.json.tmp"
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(target / "manifest.json")
