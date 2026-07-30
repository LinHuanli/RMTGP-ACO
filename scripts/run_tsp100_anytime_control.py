#!/usr/bin/env python3
"""运行 TSP100 Anytime+Final 的 3 variants × 3 seeds 匹配对照。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_2opt_signal_formal as formal  # noqa: E402


def main() -> int:
    formal.DECISIONS = (
        ROOT
        / "experiments"
        / "tsp100_2opt_anytime"
        / "formal_decisions.json"
    )
    formal.RUN_ROOT = (
        ROOT / "runs" / "tsp100-2opt-anytime" / "formal"
    )
    # 本对照不使用 Origin gate，但保留路径变量以满足共享调度器接口。
    formal.AUDIT_ROOT = (
        ROOT / "runs" / "tsp100-2opt-signal-v2" / "audit"
    )
    return formal.main()


if __name__ == "__main__":
    raise SystemExit(main())
