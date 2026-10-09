"""从现存训练文件抽取固定子集；标准测试文件全量保留且顺序不变。"""

import argparse
import json
import sys
from hashlib import sha256
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gpaco.data import ROOT, coordinate_hash, parse_line, tour_length, write_json


def save_split(n, split, records):
    target = ROOT / "Datasets/processed/v1" / f"tsp{n}" / split
    target.mkdir(parents=True, exist_ok=True)
    coords = np.stack([r[0] for r in records])
    tours = np.stack([r[1] for r in records])
    ids = np.asarray([r[2] for r in records], dtype="U64")
    arrays = {
        "coords_fp32": coords.astype(np.float32),
        "coords_original": coords,
        "reference_tours": tours,
        "instance_ids": ids,
        "reference_lengths_original": np.asarray(
            [tour_length(c, t) for c, t in zip(coords, tours, strict=True)]
        ),
    }
    hashes = {}
    for name, values in arrays.items():
        path = target / (name + ".npy")
        np.save(path, values, allow_pickle=False)
        hashes[name] = sha256(path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "n": n,
        "split": split,
        "count": len(records),
        "format": "npy",
        "distance_convention": "continuous_euclidean",
        "reference_type": "user_supplied_optimal_tour",
        "optimality_certificates_checked": False,
        "arrays": hashes,
        "instances": [{"id": r[2], "source": r[3], "line": r[4]} for r in records],
    }
    write_json(target / "manifest.json", manifest)
    write_json(ROOT / "Datasets/splits" / f"tsp{n}_{split}.json", manifest)
    print(f"saved tsp{n}/{split}: {len(records)}", flush=True)


def prepare(n, seed, train_count):
    complete = ROOT / "Datasets/processed/v1" / f"tsp{n}/COMPLETE.json"
    if complete.exists():
        payload = json.loads(complete.read_text())
        if payload["seed"] != seed or payload["train_count"] != train_count:
            raise ValueError("已有数据版本与请求不符，请创建新版本，禁止覆盖冻结划分")
        print(f"tsp{n}: 已完成，不覆盖", flush=True)
        return
    legacy = json.loads((ROOT / "Datasets/manifest.json").read_text())
    files = sorted(
        [f for f in legacy["files"] if f["split"] == "train" and f["scale"] == n],
        key=lambda f: f["path"],
    )
    test_name = {100: "tsp100_concorde_7.756.txt", 500: "tsp500_concorde_16.546.txt"}[n]
    test = ROOT / "Datasets/TSP/test_dataset/tsp" / test_name
    test_records = []
    seen = set()
    with test.open() as source:
        for row, line in enumerate(source, 1):
            coords, tour = parse_line(line)
            identity = coordinate_hash(coords)
            # 标准测试集合本身不删样本；重复只做记录。
            seen.add(identity)
            test_records.append((coords, tour, identity, str(test.relative_to(ROOT)), row))
    save_split(n, "test", test_records)
    # 其他现存测试分布只用于防泄漏检查，不输入训练、调参和模型选择。
    for path in sorted(test.parent.glob(f"tsp{n}_*.txt")):
        with path.open() as source:
            for line in source:
                coords, _ = parse_line(line)
                seen.add(coordinate_hash(coords))
    sizes = [("validation", 128), ("tuning", 128), ("holdout", 128), ("train", train_count)]
    needed = sum(count for _, count in sizes)
    total = sum(f["instances"] for f in files)
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, n])))
    selected = np.sort(rng.choice(total, size=min(total, needed + 2048), replace=False))
    selected_set = set(int(x) for x in selected)
    records, source_hashes = [], {}
    offset = 0
    for info in files:
        path = ROOT / "Datasets/TSP" / info["path"]
        digest = sha256()
        count = 0
        with path.open("rb") as source:
            for row, raw in enumerate(source, 1):
                digest.update(raw)
                if offset + row - 1 in selected_set:
                    coords, tour = parse_line(raw.decode())
                    identity = coordinate_hash(coords)
                    if identity not in seen:
                        seen.add(identity)
                        records.append((coords, tour, identity, str(path.relative_to(ROOT)), row))
                count += 1
        if count != info["instances"] or digest.hexdigest() != info["sha256"]:
            raise ValueError(f"源文件与历史清单不一致：{path}")
        source_hashes[str(path.relative_to(ROOT))] = digest.hexdigest()
        offset += count
        print(f"checked {path.name}; collected={len(records)}", flush=True)
    if len(records) < needed:
        raise ValueError("去重后样本不足；未生成 COMPLETE，不允许直接用于实验")
    permutation = rng.permutation(len(records))[:needed]
    records = [records[int(i)] for i in permutation]
    begin = 0
    for split, count in sizes:
        save_split(n, split, records[begin : begin + count])
        begin += count
    write_json(
        complete,
        {
            "schema_version": 1,
            "seed": seed,
            "train_count": train_count,
            "source_sha256": source_hashes,
            "test_file": str(test.relative_to(ROOT)),
            "test_sha256": sha256(test.read_bytes()).hexdigest(),
            "test_count": len(test_records),
            "disjoint": True,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scales", nargs="+", type=int, default=[100, 500])
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--train-count", type=int, default=32768)
    args = parser.parse_args()
    for scale in args.scales:
        prepare(scale, args.seed, args.train_count)
