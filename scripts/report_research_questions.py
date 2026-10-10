"""从分类报告生成RQ证据链，不把不同证据等级合并为一个统计样本。"""

from pathlib import Path

import numpy as np
from report_baseline_diagnostics import export, plt, save
from report_research_results import publish, read

from gpaco.data import ROOT


def report():
    out = ROOT / "docs/results/analysis/p01"
    sources = [Path(__file__).resolve()]

    def table(relative):
        path = ROOT / "docs/results" / relative
        sources.extend([path, path.parent.parent / "provenance.json"])
        return read(path)

    stages = table("diagnostic/E01/p01/tables/gpu_stage_profiles.json")
    work = table("pilot/E01/p01/tables/gpu_work_metrics.json")
    jit = table("pilot/E04/p01/tables/performance.json")
    mapping = table("pilot/E07/p01/tables/paired_summary.json")
    formal = table("formal/E09/p01/tables/summary.json")
    cross = table("pilot/E12/p01/tables/within_model_pairs.json")
    rq1, scaling = [], []
    for n in (100, 500):
        for generation in (1, 25, 50):
            values = [
                r
                for r in stages
                if (r["n"], r["generation"], r["variant"]) == (n, generation, "as")
            ]
            if values:
                rq1.append(
                    {
                        k: values[0][k]
                        for k in (
                            "n",
                            "generation",
                            "construct_percent_of_recorded_stages",
                            "construct_device_s",
                            "update_device_s",
                        )
                    }
                )
            values = [
                r
                for r in work
                if (r["n"], r["generation"], r["variant"], r["mode"])
                == (n, generation, "as", "generated")
            ]
            scaling.append(
                dict(
                    n=n,
                    generation=generation,
                    clean_records=len(values),
                    median_device_s=float(np.median([r["device_search_s"] for r in values])),
                    ns_per_transition=float(
                        np.median(
                            [
                                r["device_search_s"] / r["effective_transitions"] * 1e9
                                for r in values
                            ]
                        )
                    ),
                    fallback_event_percent=float(
                        np.median([r["fallback_event_percent"] for r in values])
                    ),
                    sampled_gpu_gib=float(
                        np.median([r["sampled_gpu_used_bytes"] / 2**30 for r in values])
                    ),
                )
            )
    export(out, "rq1_as_stages", rq1)
    export(out, "rq4_as_scaling", scaling)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for n in (100, 500):
        values = [r for r in scaling if r["n"] == n]
        axes[0].plot(
            [r["generation"] for r in values],
            [r["ns_per_transition"] for r in values],
            "o-",
            label=f"TSP{n}",
        )
        axes[1].plot(
            [r["generation"] for r in values],
            [r["fallback_event_percent"] for r in values],
            "o-",
            label=f"TSP{n}",
        )
    for ax in axes:
        ax.set_xlabel("Real cohort generation")
        ax.set_xticks([1, 25, 50])
        ax.legend()
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Device ns per constructed transition")
    axes[1].set_ylabel("Fallback events / constructed transitions (%)")
    save(fig, out, "rq4_scale_and_cohort")
    lines = [
        "本页是证据导航与分析，不是将pilot、formal和diagnostic合并成一个统计样本。所有质量指标为同FP32几何的标签gap%；标准测试未打开。",
        "",
        "## RQ1：主要成本在哪里？",
        "",
        "AS阶段事件如下。三个cohort分别测量，不对单次profiling计算置信区间。",
        "",
        "| n | cohort代 | 构造秒 | 更新秒 | 构造占已记录阶段比例 |",
        "|---|---:|---:|---:|---:|",
    ]
    lines += [
        f"| {r['n']} | {r['generation']} | {r['construct_device_s']:.2f} | {r['update_device_s']:.2f} | {r['construct_percent_of_recorded_stages']:.2f}% |"
        for r in rq1
    ]
    lines += [
        "",
        "构造包含特征、树求值、选边及同步ACS局部更新，不能把整项称为GP时间。54项细粒度诊断进一步区分候选扫描、排名、统计和lane采样周期；见[按规模/宿主分面的诊断图](../../diagnostic/E01/p01/figures/detailed_work_and_sampled_cycles.svg)。这支持优先分析构造中的工作，而不是证明某个未实施优化有效。SM occupancy、DRAM、stall和spill流量缺失，不能以NVML利用率代替。",
        "",
        "## RQ2：现有执行加速成立到什么程度？",
        "",
        "| 对照 | 当前证据 | 允许的结论 |",
        "|---|---|---|",
        "| Python 1/8/16核 | 代码及功能检查已准备，完整计时缺失 | 暂不报告GPU/CPU倍数 |",
        "| Numba 1/8/16核 | 同上 | 暂不判断CPU扩展或Numba收益 |",
        "| GPU解释器/树JIT | 90预设配对block；3个争用block整体排除 | 可以比较当前GPU树专门化，不是CPU/新方法收益 |",
        "| CPU-Opt、GPU-Tensor、M1/M2/M3 | 未完成 | 不填推测值 |",
        "",
        "AS的同卡配对结果：",
        "",
        "| n | cohort代 | 干净block | JIT秒中位数 | 解释器/JIT配对中位比 |",
        "|---|---:|---:|---:|---:|",
    ]
    lines += [
        f"| {r['n']} | {r['generation']} | {r['clean_blocks']}/5 | {r['jit_median_s']:.2f} | {r['paired_ratio']:.3f} |"
        for r in jit
        if r["variant"] == "as"
    ]
    lines += [
        "",
        "完整三宿主结果与探索性配对区间见[E04](../../pilot/E04/p01/README.md)。时间为完整预算的暖评价，编译预热另列，不是完整训练加速比。",
        "",
        "## RQ3：怎样组织GPU映射？",
        "",
        "固定active=3200时，相对lanes8的配对时间比如下；大于1表示被比较计划更快。这里只展示预定固定计划，不在holdout选正式赢家。",
        "",
        "| n | cohort代 | lanes4 比值 | lanes16 比值 | lanes32 |",
        "|---|---:|---:|---:|---|",
    ]
    for n in (100, 500):
        for g in (1, 25, 50):
            rows = {
                r["lanes"]: r
                for r in mapping
                if (r["n"], r["generation"], r["active"]) == (n, g, 3200)
            }
            lines.append(
                f"| {n} | {g} | {rows[4]['reference_over_plan']:.3f} | {rows[16]['reference_over_plan']:.3f} | 寄存器资源下不可启动 |"
            )
    lines += [
        "",
        "[TSP100热图](../../pilot/E07/p01/figures/tsp100_fixed_mapping.svg)及[TSP500热图](../../pilot/E07/p01/figures/tsp500_fixed_mapping.svg)显示映射与cohort存在交互。增大每蚂蚁线程数不保证更快；不可行也不能记为零时间。两个污染a01保留，独立修复队列的a02只能替代同一block，不能增加重复数。这不是完整M3选择器验证。",
        "",
        "## RQ4：TSP500为什么更慢？",
        "",
        "| n | cohort代 | 搜索秒中位数 | ns/构造transition | fallback事件% | 采样显存GiB |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    lines += [
        f"| {r['n']} | {r['generation']} | {r['median_device_s']:.2f} | {r['ns_per_transition']:.2f} | {r['fallback_event_percent']:.2f} | {r['sampled_gpu_gib']:.2f} |"
        for r in scaling
    ]
    lines += [
        "",
        "![规模与cohort](figures/rq4_scale_and_cohort.png)",
        "",
        "n从100到500，单条路径的构造步数从99到499，密集状态矩阵容量约随n²增长。每transition成本、回退率和cohort也在变化，因此不能只用5倍路径长度解释总耗时。AS初代JIT收益较小，后期收益明显增大；TSP500并非始终比TSP100的相对收益小。后期两规模的GP已经分别进化，这不是只改变n的纯因果干预。当前证据支持继续做固定程序/状态、资源布局和排名工作干预，尚不能断言唯一原因是显存带宽。正式seed2004的慢运行保留，不因它慢就删除。旧双树/FP64历史结果另列[E02](../../historical/E02/p01/README.md)，不作直接配对分母。",
        "",
        "## RQ5：更快是否让算法设计更好？",
        "",
        "当前能回答固定GPU对照能否学到有效规则，不能回答新加速方法的等时间收益。正式TSP100的10根seed结果如下。",
        "",
        "| 指标 | 均值 | 根seed SD | 95%根seed bootstrap区间 |",
        "|---|---:|---:|---|",
    ]
    for r in formal:
        if r["n"] == 100:
            lines.append(
                f"| {r['metric']} | {r['mean']:.4f} | {r['sd']:.4f} | [{r['ci_low']:.4f}, {r['ci_high']:.4f}] |"
            )
    lines += [
        "",
        "![正式TSP100训练](../../formal/E09/p01/figures/tsp100_training.png)",
        "",
        "训练展示本代最好和中位数，并保留随batch波动的ACO对照；验证展示规定代的已选冠军，不补造中间验证点。验证改善不能代替最终测试，也不能消除反复验证选择的偏差。TSP500是否齐全见[正式覆盖表](../../formal/E09/p01/README.md)，不对未完成集合发布总体均值。三seed先导单列，不能与正式seed相加。",
        "",
        "## RQ6：能迁移到不同宿主和硬件吗？",
        "",
        "| 型号 | n | 干净block | 默认秒 | 独立调优后秒 | 默认/调优配对比 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    lines += [
        f"| {r['model']} | {r['n']} | {r['clean_blocks']}/5 | {r['default_s']:.2f} | {r['selected_s']:.2f} | {r['default_over_selected']:.3f} |"
        for r in cross
    ]
    lines += [
        "",
        "[跨GPU图与资源、能耗、逐位一致性、统一冠军审计表](../../pilot/E12/p01/README.md)已分开导出。每型号只有一张实际卡，不能解释为型号总体的独立硬件重复。FP32跨架构可能产生不同闭环路径；规范审计不重新选择冠军。现有三宿主性能是固定AS程序跨宿主执行，不等于三宿主独立训练都有效。",
        "",
        "## p02将补充什么？",
        "",
        "新增2-opt/3-opt执行器同语义对照、构造/LS/更新成本，以及三宿主各自联合训练。问题是执行速度及GP相对于同LS的ZERO基准是否改善，不预设结果必然正向。12配置先通过资格、调优和三代smoke，再放行36次50代pilot。标准测试、10seed LS确认性实验和等时间政策仍未解锁。",
    ]
    publish(out, "现有结果：六个RQ的证据与边界", "synthesis: see each source tier", lines, sources)


if __name__ == "__main__":
    report()
