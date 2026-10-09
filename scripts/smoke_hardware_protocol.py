"""仅验证跨卡流水线控制流；缩小 ACO 预算，产物不能用于科研速度/质量结论。"""

import argparse
import json
import os

import yaml

from gpaco.config import SearchConfig
from gpaco.data import ROOT, write_json
from gpaco.experiment import train
from gpaco.hardware_campaign import audit_available, hardware_info, holdout, tune
from gpaco.hardware_inputs import FrozenStore, TrainingInputs, prepare_scale, safe_directory
from gpaco.telemetry import Monitor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    campaign = safe_directory(args.output)
    campaign.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load((ROOT / "configs/hardware/cross_gpu_pilot.yaml").read_text())
    config.update(
        sizes=[100],
        seeds=[1001],
        generations=1,
        batch=2,
        validation_repeats=1,
        paired_blocks=2,
        name=str(campaign.relative_to(ROOT / "artifacts")),
    )
    config["search"].update(ants=4, iterations=3)
    config["targets"] = {"a5000": config["targets"]["a5000"]}
    uuid = config["targets"]["a5000"]["gpu_uuid"]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != uuid:
        raise RuntimeError("smoke 需要指定的空闲 A5000，不自动选择设备")
    write_json(campaign / "campaign.json", {"config": config, "smoke_only": True})
    device = campaign / "devices/a5000"
    device.mkdir(parents=True)
    monitor = Monitor(uuid, device)
    try:
        prepare_scale(campaign / "inputs/tsp100", 100, config, uuid)
        store = FrozenStore(campaign / "inputs/tsp100")
        hardware = hardware_info()
        write_json(device / "hardware.json", hardware)
        target = device / "tsp100"
        search = SearchConfig(**config["search"])
        plan = tune(
            target / "tuning",
            store,
            search,
            config,
            hardware["device_properties"]["multiProcessorCount"],
            monitor,
        )
        holdout(target / "holdout", store, search, plan, config, "a5000", 100, monitor)
        train(
            target / "training/seed-1001",
            100,
            1001,
            search,
            plan,
            population_size=100,
            generations=1,
            batch_size=2,
            validation_repeats=1,
            inputs=TrainingInputs(store, 1001),
        )
        audit_available(campaign, store, config, monitor)
        write_json(
            campaign / "SMOKE_COMPLETE.json", {"status": "passed", "scientific_sample": False}
        )
        print(json.dumps({"status": "smoke_passed", "output": str(campaign)}), flush=True)
    finally:
        monitor.close()


if __name__ == "__main__":
    main()
