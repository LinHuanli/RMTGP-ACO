"""跨实现的内容校验输入包；显式导入旧实验，不放宽 FrozenStore 的版本保护。"""

import json
import shutil
from dataclasses import asdict

import numpy as np

from .config import SearchConfig
from .data import ROOT, FrozenInitialization, ProblemSpec, write_json
from .hardware_inputs import ARRAYS, file_hash, safe_directory
from .language import ProgramSpec

SCHEMA = "gpaco-benchmark-inputs-v1"


def contract_hash():
    return file_hash(ROOT / "configs/semantic_contract.yaml")


def export_bundle(source, cohorts, output):
    """只复制校验过的冻结字节；不重新计算几何、初态或随机种子。"""
    source, output = safe_directory(source), safe_directory(output)
    manifest = json.loads((source / "manifest.json").read_text())
    identity = file_hash(source / "manifest.json")
    if identity != json.loads((source / "READY.json").read_text())["manifest_sha256"]:
        raise ValueError("原始输入 manifest 被改动")
    if manifest["writer"]["semantic_contract_hash"] != contract_hash():
        raise ValueError("原始输入不属于当前语义合同，不能导入主对照")
    if manifest.get("tests_opened") or manifest["geometry"]["holdout"]["source_split"] != "holdout":
        raise ValueError("仅允许性能 holdout，不允许开启标准测试集")
    files = {}
    for relative, expected in manifest["files"].items():
        path = (source / relative).resolve()
        if not path.is_relative_to(source) or file_hash(path) != expected:
            raise ValueError(f"源输入校验失败：{relative}")
    cohort_rows = {}
    cohort_sources = {}
    for generation, path in cohorts.items():
        path = safe_directory(path)
        rows = json.loads(path.read_text())
        programs = [ProgramSpec.parse(r["expression"]) for r in rows]
        if any(p.semantic_hash != r["semantic_hash"] for p, r in zip(programs, rows, strict=True)):
            raise ValueError("程序文本与哈希不匹配")
        cohort_rows[str(generation)] = [p.record() for p in programs]
        cohort_sources[str(generation)] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": file_hash(path),
        }
    output.mkdir(parents=True, exist_ok=False)
    for relative, expected in manifest["files"].items():
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, target)
        if file_hash(target) != expected:
            raise ValueError("输入复制校验失败")
        files[relative] = expected
    for generation, rows in cohort_rows.items():
        relative = f"cohorts/g{int(generation):03d}.json"
        write_json(output / relative, rows)
        files[relative] = file_hash(output / relative)
    result = {
        "schema": SCHEMA,
        "semantic_contract_sha256": contract_hash(),
        "source_manifest_sha256": identity,
        "source_hash": manifest["source_hash"],
        "search": manifest["search"],
        "geometry": manifest["geometry"],
        "scenarios": manifest["scenarios"],
        "files": files,
        "cohorts": cohort_sources,
        "tests_opened": False,
        "source_directory": str(source.relative_to(ROOT)),
        "provenance_note": "显式跨实现导入；源码身份只作来源记录，文件字节和语义合同必须核验",
    }
    write_json(output / "manifest.json", result)
    write_json(output / "READY.json", {"manifest_sha256": file_hash(output / "manifest.json")})
    return result


class BenchmarkInputs:
    """只读、相对路径、无 GPU 依赖；适用于 CPU 基线和新的 GPU 诊断。"""

    def __init__(self, directory):
        self.directory = safe_directory(directory)
        self.identity = file_hash(self.directory / "manifest.json")
        ready = json.loads((self.directory / "READY.json").read_text())
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        if ready["manifest_sha256"] != self.identity or self.manifest["schema"] != SCHEMA:
            raise ValueError("输入包未完成或格式错误")
        if self.manifest["semantic_contract_sha256"] != contract_hash():
            raise ValueError("语义合同不匹配")
        self.search = SearchConfig(**self.manifest["search"])
        self.verified = set()

    def path(self, relative):
        target = (self.directory / relative).resolve()
        if not target.is_relative_to(self.directory):
            raise ValueError("输入路径越界")
        if relative not in self.verified:
            if file_hash(target) != self.manifest["files"][relative]:
                raise ValueError(f"输入文件被改动：{relative}")
            self.verified.add(relative)
        return target

    def load(self, generation, block):
        if str(generation) not in self.manifest["cohorts"]:
            raise FileNotFoundError(f"真实 generation {generation} cohort 尚未导入")
        rows = json.loads(self.path(f"cohorts/g{generation:03d}.json").read_text())
        programs = [ProgramSpec.parse(r["expression"]) for r in rows]
        key = f"block-{block:02d}"
        scenario = self.manifest["scenarios"][key]
        geometry = self.manifest["geometry"][scenario["geometry"]]
        prefix = geometry["path"]
        arrays = {
            name: np.load(self.path(f"{prefix}/{name}.npy"), mmap_mode="r", allow_pickle=False)
            for name in ARRAYS
        }
        ids = tuple(json.loads(self.path(f"{prefix}/ids.json").read_text()))
        if ids != tuple(geometry["instances"]):
            raise ValueError("实例顺序不匹配")
        if any(
            arrays[f].dtype != np.float32
            for f in ("coords", "distances", "heuristic", "log_heuristic", "reference")
        ):
            raise ValueError("搜索浮点数据必须为 FP32")
        initial = tuple(
            np.load(self.path(f"scenarios/{key}/{f}.npy"), mmap_mode="r", allow_pickle=False)
            for f in ("tau0", "low", "high")
        )
        frozen = FrozenInitialization(
            scenario["seed"],
            self.search.variant,
            self.search.rho,
            ids,
            initial,
            key,
            self.manifest["source_manifest_sha256"],
        )
        return programs, ProblemSpec(**arrays, instance_ids=ids, initialization=frozen)

    def workload(self, generation, block, programs, problem):
        return {
            "input_manifest_sha256": self.manifest["source_manifest_sha256"],
            "bundle_sha256": self.identity,
            "generation": generation,
            "block": block,
            "seed": problem.initialization.seed,
            "search": asdict(self.search),
            "program_hashes": [p.semantic_hash for p in programs],
            "n": problem.n,
            "p": len(programs),
            "b": problem.size,
            "requested_tasks": len(programs) * problem.size,
            "fitness_cache_hits": 0,
        }
