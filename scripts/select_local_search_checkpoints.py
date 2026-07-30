#!/usr/bin/env python3
"""在统一 ACO+2opt validation 环境中重新选择一组 GP checkpoints。"""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict, replace
from pathlib import Path

from rmtgp_aco.baseline import BaselineArchive
from rmtgp_aco.config import ACOVariant, LocalSearch
from rmtgp_aco.evaluation import load_champion
from rmtgp_aco.sampling import pools_from_paths
from rmtgp_aco.schedule import load_schedule, validation_cases_from_schedule
from rmtgp_aco.spec import load_run_spec
from rmtgp_aco.training import (
    BaselineCache,
    EvaluationPool,
    _cpu_fp64_final_audit,
    validate_candidates,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    ROOT / "experiments" / "tsp100_local_search_3seed" / "config.yaml"
)
TUNING = ROOT / "configs/cuda_tuning/rtx_pro5000_blackwell_sm120_v1.json"


def _experiment(config: Path, variant: ACOVariant, seed: int):
    spec = load_run_spec(config)
    parameters = {
        ACOVariant.AS: {"rho": 0.5, "q0": 0.0},
        ACOVariant.ACS: {"rho": 0.1, "q0": 0.98},
        ACOVariant.MMAS: {"rho": 0.2, "q0": 0.0},
    }[variant]
    aco = replace(
        spec.experiment.aco,
        variant=variant,
        ants=32,
        rho=parameters["rho"],
        q0=parameters["q0"],
        local_search=LocalSearch.TWO_OPT,
        iterations=500,
    )
    runtime = replace(
        spec.experiment.runtime,
        gpu_devices=(0,),
        cuda_tuning_manifest=str(TUNING),
    )
    return spec, replace(
        spec.experiment,
        experiment_id=f"tsp100-ls-select-{variant.value}-seed-{seed}",
        root_seed=seed,
        aco=aco,
        runtime=runtime,
    )


def _atomic_pickle(path: Path, value: object) -> None:
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--variant", choices=[item.value for item in ACOVariant], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--replicate-id", type=int, default=0)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--baseline-archive", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--cpu-audit",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    variant = ACOVariant(args.variant)
    spec, experiment = _experiment(args.config, variant, args.seed)
    schedule = load_schedule(args.schedule)
    validation_pools = pools_from_paths(spec.data.validation_paths())
    common = {
        "replicate_id": args.replicate_id,
        "batch_size": spec.data.evaluation_batch_size,
        "candidate_size": experiment.aco.candidate_size,
        "dtype": experiment.aco.dtype,
        "device": experiment.aco.device,
    }
    screening = validation_cases_from_schedule(
        schedule,
        validation_pools,
        role="selection",
        seeds=experiment.validation_screening_seeds,
        **common,
    )
    selection_cases = validation_cases_from_schedule(
        schedule,
        validation_pools,
        role="selection",
        seeds=experiment.validation_seeds,
        **common,
    )
    gate = validation_cases_from_schedule(
        schedule,
        validation_pools,
        role="gate",
        seeds=experiment.validation_seeds,
        **common,
    )
    checkpoint_paths = sorted((args.run / "checkpoints").glob("candidate_*.pkl"))
    if not checkpoint_paths:
        raise FileNotFoundError(f"没有找到 checkpoints: {args.run}")
    candidates = [load_champion(path) for path in checkpoint_paths]
    archive = BaselineArchive(
        args.baseline_archive,
        experiment.aco,
        experiment.runtime.aco_backend,
        require=True,
        runtime=experiment.runtime,
    )
    cache = BaselineCache(archive)
    with EvaluationPool(experiment) as evaluator:
        evaluator.warm(screening[0])
        selection = validate_candidates(
            candidates,
            experiment,
            selection_cases,
            cache,
            screening_cases=screening,
            gate_cases=gate,
            evaluator_pool=evaluator,
        )

    audit = (
        _cpu_fp64_final_audit(
            selection.selected_candidate,
            experiment,
            gate,
        )
        if args.cpu_audit
        else None
    )
    target = args.output or (args.run / "selected_candidate_2opt.pkl")
    target.parent.mkdir(parents=True, exist_ok=True)
    _atomic_pickle(target, selection.selected_candidate)
    payload = {
        "schema_version": 1,
        "variant": variant.value,
        "gp_seed": args.seed,
        "selection_environment": {
            "local_search": "two_opt",
            "ants": 32,
            "iterations": 500,
            "candidate_size": 20,
        },
        "checkpoint_count": len(candidates),
        "selected_candidate_hash": selection.selected_candidate_hash,
        "selected_macro_gap_percent": selection.selected_macro_gap_percent,
        "selected_macro_delta_pp": selection.selected_macro_delta_pp,
        "passed_noninferiority": selection.passed_noninferiority,
        "scales": [asdict(item) for item in selection.scales],
        "cpu_fp64_audit": (
            {
                "selected_macro_gap_percent": audit.selected_macro_gap_percent,
                "selected_macro_delta_pp": audit.selected_macro_delta_pp,
                "passed_noninferiority": audit.passed_noninferiority,
                "scales": [asdict(item) for item in audit.scales],
            }
            if audit is not None
            else None
        ),
        "output": str(target.resolve()),
    }
    summary = target.with_suffix(".json")
    temporary = summary.with_suffix(f"{summary.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(summary)
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
