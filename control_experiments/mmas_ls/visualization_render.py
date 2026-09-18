"""生成机制解释的中文报告、论文图和离线交互页面。"""
# ruff: noqa: E501

from __future__ import annotations

import base64
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .common import atomic_json, file_hash, now, read_json
from .mechanism_visualization import BEHAVIORS, REGIMES, SAMPLE_ITERATIONS
from .report_render import markdown_table, pyplot, save


def _pyplot():
    """本机存在的中文字体；避免论文图中的汉字缺字。"""
    plt = pyplot()
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Droid Sans Fallback", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    )
    return plt


def _read(path: Path) -> list[dict]:
    with Path(path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def _f(row: dict, key: str) -> float:
    return float(row[key])


def _summary(path: Path, metric: str, **filters) -> list[dict]:
    return [
        row
        for row in _read(path)
        if row["metric"] == metric
        and all(str(row[key]) == str(value) for key, value in filters.items())
    ]


def plot_source_policy(report: Path) -> None:
    rows = _read(report / "source_policy_performance.csv")
    own = [
        row
        for row in rows
        if row["execution_configuration"].startswith(row["expression_origin"].split()[0])
    ]
    labels = [row["execution_configuration"].replace("：", "\n") for row in own]
    x = np.arange(len(rows := own))
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), layout="constrained")
    axes[0].plot(x, [_f(row, "baseline_gap_percent") for row in rows], "o-", label="不使用 GP")
    axes[0].plot(x, [_f(row, "gp_gap_percent") for row in rows], "s-", label="使用同框架训练的 GP")
    axes[0].set(
        xticks=x, xticklabels=labels, ylabel="reference gap（%）", title="来源策略改变绝对质量"
    )
    axes[0].tick_params(axis="x", labelrotation=25)
    axes[0].legend()
    benefit = np.array([_f(row, "gp_benefit_pp") for row in rows])
    axes[1].bar(x, benefit, color=np.where(benefit >= 0, "#2673b8", "#c64949"))
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set(
        xticks=x,
        xticklabels=labels,
        ylabel="GP 改进（百分点；正值更好）",
        title="来源越集中于优良路径，GP 的边际空间越小",
    )
    axes[1].tick_params(axis="x", labelrotation=25)
    save(plt, fig, report / "source_policy_performance")


def plot_same_state_transfer(report: Path) -> None:
    source_path = report / "same_state_source_support_summary.csv"
    probability_path = report / "same_state_probability_summary.csv"
    update_path = report / "same_state_update_tree_summary.csv"
    source = _read(source_path)
    probability = _read(probability_path)
    update = _read(update_path)
    plt = _pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), layout="constrained")
    categories = (
        "仅历史来源路径包含的边",
        "仅本轮最优路径包含的边",
        "两条来源路径共有边",
        "其他候选弧",
    )
    colors = ("#2673b8", "#c64949", "#6b6b6b", "#c2c2c2")
    for category, color in zip(categories, colors, strict=True):
        points = [
            row
            for row in source
            if row["metric"] == "mean_signed_tau_change"
            and row["state_owner"] == "不使用 GP 产生的状态"
            and row["historical_source"] == "历史全局最优路径"
            and row["update_rule"] == "来源内均匀沉积"
            and row["phase"] == "沉积后"
            and row["edge_category"] == category
        ]
        points.sort(key=lambda row: int(row["snapshot_iteration"]))
        axes[0].plot(
            [int(row["snapshot_iteration"]) for row in points],
            [_f(row, "mean") for row in points],
            "o-",
            label=category,
            color=color,
        )
    axes[0].axhline(0, color="black", lw=0.7)
    axes[0].set(
        xscale="log",
        xlabel="快照轮次",
        ylabel="历史来源替代后的平均信息素变化",
        title="改变来源先改变沉积支持集",
    )
    axes[0].legend(fontsize=7)
    for affected, label, color in (
        ("affected_probability_tv", "候选集含来源差异边", "#2673b8"),
        ("unaffected_probability_tv", "候选集不含来源差异边", "#888888"),
    ):
        points = [
            row
            for row in probability
            if row["metric"] == affected
            and row["state_owner"] == "不使用 GP 产生的状态"
            and row["historical_source"] == "历史全局最优路径"
            and row["update_rule"] == "来源内均匀沉积"
        ]
        points.sort(key=lambda row: int(row["snapshot_iteration"]))
        axes[1].plot(
            [int(row["snapshot_iteration"]) for row in points],
            [_f(row, "mean") for row in points],
            "o-",
            label=label,
            color=color,
        )
    axes[1].set(
        xscale="log",
        xlabel="快照轮次",
        ylabel="下一步选择概率 TV",
        title="支持集变化只在相关决策上下文中传播",
    )
    axes[1].legend(fontsize=8)
    for source_name, color in (("本轮最优路径", "#555555"), ("历史全局最优路径", "#2673b8")):
        points = [
            row
            for row in update
            if row["metric"] == "source_edge_tau_l1_over_two_budget"
            and row["state_owner"] == "不使用 GP 产生的状态"
            and row["source"] == source_name
            and row["phase"] in ("更新前", "蒸发和下界保护后", "沉积后", "重启和可选上界处理后")
        ]
        grouped = {}
        for row in points:
            grouped.setdefault(row["phase"], []).append(_f(row, "mean"))
        axes[2].plot(
            np.arange(4),
            [
                np.mean(grouped.get(phase, [np.nan]))
                for phase in ("更新前", "蒸发和下界保护后", "沉积后", "重启和可选上界处理后")
            ],
            "o-",
            label=source_name,
            color=color,
        )
    axes[2].set(
        xticks=np.arange(4),
        xticklabels=["更新前", "蒸发后", "沉积后", "重启后"],
        ylabel="GP 引起的来源边信息素 L1 / 2B",
        title="GP 只在已选来源内部重分配固定预算",
    )
    axes[2].legend(fontsize=8)
    save(plt, fig, report / "same_state_tau_transfer")


def plot_pheromone_trajectory(report: Path) -> None:
    rows = _read(report / "pheromone_trajectory_summary.csv")
    plt = _pyplot()
    panels = (
        ("candidate_effective_arcs", "有效候选弧数"),
        ("candidate_gini", "候选弧信息素 Gini"),
        ("top500_mass", "最高 500 条候选弧质量占比"),
        ("source_edge_lift", "实际强化路径边的信息素提升倍数"),
        ("reference_edge_lift", "参考路径边的信息素提升倍数"),
        ("post_unique_tours", "2-opt 后不同路径数"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    for ax, (metric, label) in zip(axes.flat, panels, strict=True):
        for behavior, color in (("不使用 GP", "#222222"), ("三个固定 GP 表达式的均值", "#2673b8")):
            points = [
                row for row in rows if row["metric"] == metric and row["behavior"] == behavior
            ]
            points.sort(key=lambda row: int(row["iteration"]))
            x = np.array([int(row["iteration"]) for row in points])
            mean = np.array([_f(row, "mean") for row in points])
            low = np.array([_f(row, "point_low") for row in points])
            high = np.array([_f(row, "point_high") for row in points])
            ax.plot(x, mean, label=behavior, color=color, lw=1)
            ax.fill_between(x, low, high, color=color, alpha=0.12)
        ax.set(xscale="log", xlabel="ACO 轮次", ylabel=label)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle("完整 MMAS 中的信息素集中、路径对齐和群体多样性")
    save(plt, fig, report / "pheromone_concentration")


def plot_response_and_local_search(report: Path) -> None:
    quality = _read(report / "continuation_quality_across_snapshots_summary.csv")
    structure = _read(report / "two_opt_survival_across_snapshots_summary.csv")
    plt = _pyplot()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), layout="constrained")
    for metric, label, color in (
        ("原生来源下 GP 增益（百分点）", "原生历史来源", "#222222"),
        ("仅本轮最优来源下 GP 增益（百分点）", "移除历史来源", "#2673b8"),
    ):
        points = [
            row
            for row in quality
            if row["metric"] == metric
            and row["state_owner"] == "不使用 GP 产生的状态"
        ]
        points.sort(key=lambda row: int(row["continuation_iterations"]))
        axes[0].plot(
            [int(row["continuation_iterations"]) for row in points],
            [_f(row, "mean") for row in points],
            "o-",
            label=label,
            color=color,
        )
    axes[0].axhline(0, color="black", lw=0.7)
    axes[0].set(
        xscale="log",
        xlabel="同状态续跑轮数",
        ylabel="GP 改进（百分点）",
        title="即时相同，反馈累积后分离",
    )
    axes[0].legend(fontsize=8)
    for metric, label, color in (
        ("仅构造决策树的增益（百分点）", "仅构造树", "#d98b2b"),
        ("仅信息素更新树的增益（百分点）", "仅更新树", "#5a9c52"),
        ("原生来源下 GP 增益（百分点）", "双树", "#2673b8"),
    ):
        points = [
            row
            for row in quality
            if row["metric"] == metric
            and row["state_owner"] == "不使用 GP 产生的状态"
        ]
        points.sort(key=lambda row: int(row["continuation_iterations"]))
        axes[1].plot(
            [int(row["continuation_iterations"]) for row in points],
            [_f(row, "mean") for row in points],
            "o-",
            label=label,
            color=color,
        )
    axes[1].axhline(0, color="black", lw=0.7)
    axes[1].set(
        xscale="log", xlabel="同状态续跑轮数", ylabel="改进（百分点）", title="两棵树的有限时域贡献"
    )
    axes[1].legend(fontsize=8)
    for metric, label, color in (
        ("pre_symmetric_difference", "2-opt 前", "#c64949"),
        ("post_symmetric_difference", "2-opt 后", "#2673b8"),
    ):
        points = [
            row
            for row in structure
            if row["metric"] == metric
            and row["state_owner"] == "不使用 GP 产生的状态"
        ]
        points.sort(key=lambda row: int(row["continuation_iterations"]))
        axes[2].plot(
            [int(row["continuation_iterations"]) for row in points],
            [_f(row, "mean") for row in points],
            "o-",
            label=label,
            color=color,
        )
    axes[2].set(
        xscale="log",
        xlabel="同状态续跑轮数",
        ylabel="每只蚂蚁的边集对称差",
        title="2-opt 消除部分构造差异，但未全部消除",
    )
    axes[2].legend(fontsize=8)
    save(plt, fig, report / "same_state_response")


def plot_historical_events(report: Path) -> None:
    rows = _read(report / "historical_source_events_summary.csv")
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    phases = ("source_lift_0", "source_lift_1", "source_lift_2", "source_lift_3")
    for behavior, color in (("不使用 GP", "#222222"), ("三个固定 GP 表达式的均值", "#2673b8")):
        values = []
        for metric in phases:
            selected = [
                row for row in rows if row["behavior"] == behavior and row["metric"] == metric
            ]
            values.append(np.mean([_f(row, "mean") for row in selected]))
        axes[0].plot(np.arange(4), values, "o-", label=behavior, color=color)
    axes[0].set(
        xticks=np.arange(4),
        xticklabels=["更新前", "蒸发后", "沉积后", "重启后"],
        ylabel="历史来源边相对候选弧均值的倍数",
        title="一次历史路径强化的即时作用",
    )
    axes[0].legend(fontsize=8)
    for behavior, color in (("不使用 GP", "#222222"), ("三个固定 GP 表达式的均值", "#2673b8")):
        overlap = [
            row
            for row in rows
            if row["behavior"] == behavior and row["metric"] == "source_current_overlap"
        ]
        diversity = [
            row
            for row in rows
            if row["behavior"] == behavior and row["metric"] == "post_unique_tours"
        ]
        axes[1].scatter(
            np.mean([_f(row, "mean") for row in overlap]),
            np.mean([_f(row, "mean") for row in diversity]),
            label=behavior,
            color=color,
            s=55,
        )
    axes[1].set(
        xlabel="历史来源与本轮最优的边重合率",
        ylabel="2-opt 后不同路径数",
        title="来源重合与群体多样性的联合位置",
    )
    axes[1].legend(fontsize=8)
    save(plt, fig, report / "historical_reinforcement_events")


def plot_expression_behavior(report: Path) -> None:
    expressions = _read(report / "expression_behavior.csv")
    raw = _read(report / "same_state_update_tree_raw.csv")
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    seeds = [int(row["seed"]) for row in expressions]
    values = []
    for seed in seeds:
        selected = [
            row
            for row in raw
            if int(row["champion"]) == seed
            and row["source"] == "历史全局最优路径"
            and row["phase"] == "沉积后"
        ]
        # 先按实例平均重复状态与 ACO seeds，再显示表达式描述值。
        by_instance = defaultdict(list)
        for row in selected:
            by_instance[int(row["instance"])].append(_f(row, "source_edge_tau_l1_over_two_budget"))
        values.append(np.mean([np.mean(v) for v in by_instance.values()]))
    axes[0].bar(np.arange(3), values, color=["#888888", "#2673b8", "#888888"])
    axes[0].set(
        xticks=np.arange(3),
        xticklabels=seeds,
        xlabel="MMAS GP 训练种子",
        ylabel="历史来源边信息素变化 / 2B",
        title="三个更新树的实际作用幅度",
    )
    transition = [row["transition_tree_active"] == "True" for row in expressions]
    pheromone = [row["pheromone_tree_active"] == "True" for row in expressions]
    axes[1].imshow(
        np.array([transition, pheromone], dtype=int), cmap="Blues", vmin=0, vmax=1, aspect="auto"
    )
    axes[1].set(
        xticks=np.arange(3),
        xticklabels=seeds,
        yticks=[0, 1],
        yticklabels=["构造决策树", "信息素更新树"],
        title="表达式是否为非零树",
    )
    for y in range(2):
        for x in range(3):
            axes[1].text(
                x, y, "非零" if [transition, pheromone][y][x] else "零树", ha="center", va="center"
            )
    save(plt, fig, report / "expression_behavior")


def _pack_dashboard(output: Path, report: Path) -> dict | None:
    if read_json(output / "replay/summary.json", {}).get("status") != "completed":
        return None
    selection = read_json(output / "selection.json")
    geometry = output / "replay/geometry.npz"
    with np.load(geometry, allow_pickle=False) as archive:
        coords = archive["coords"].astype(float)
        nearest = archive["nearest"].astype(np.int64)
    source_input = output.parent / "mechanism-explanation-v2/inputs/diagnosis_dev.npz"
    with np.load(source_input, allow_pickle=False) as archive:
        reference = archive["reference_tour"][selection["instance"]].astype(np.int64)
    order = reference[:-1]
    inverse = np.empty(500, dtype=np.int64)
    inverse[order] = np.arange(500)
    block = 25
    payload = {
        "coords": np.round(coords, 6).tolist(),
        "reference": reference.tolist(),
        "iterations": list(SAMPLE_ITERATIONS),
        "regimes": {key: value[0] for key, value in REGIMES.items()},
        "behaviors": BEHAVIORS,
        "grid": 20,
        "top": 60,
        "data": {},
    }
    for regime in REGIMES:
        payload["data"][regime] = {}
        for behavior in BEHAVIORS:
            directory = output / "replay" / regime / behavior / "samples"
            index = read_json(directory / "index.json")
            heat = bytearray()
            edges = bytearray()
            weights = bytearray()
            tours = bytearray()
            ranges = []
            metrics = []
            for name, _record in sorted(
                index["files"].items(), key=lambda item: item[1]["iteration"]
            ):
                with np.load(directory / name, allow_pickle=False) as archive:
                    tau = archive["tau"].astype(np.float32)
                    source = archive["source_tours"][0].astype(np.uint16)
                    current = archive["post_tours"][
                        int(np.argmin(archive["colony_lengths"]))
                    ].astype(np.uint16)
                    best = archive["best_tour"].astype(np.uint16)
                    trace = archive["trace"].astype(float)
                for phase in range(4):
                    reordered = np.log(np.maximum(tau[phase][order][:, order], 1e-30))
                    reduced = reordered.reshape(20, block, 20, block).mean((1, 3))
                    lo, hi = float(reduced.min()), float(reduced.max())
                    encoded = np.rint(255 * (reduced - lo) / max(hi - lo, 1e-30)).astype(np.uint8)
                    heat.extend(encoded.tobytes())
                    ranges.append([lo, hi])
                    values = tau[phase][np.arange(500)[:, None], nearest]
                    flat = np.argpartition(values.ravel(), -60)[-60:]
                    flat = flat[np.argsort(values.ravel()[flat])[::-1]]
                    u = flat // nearest.shape[1]
                    v = nearest[u, flat % nearest.shape[1]]
                    endpoints = np.stack((u, v), axis=1).astype("<u2")
                    local = values.ravel()[flat].astype(float)
                    strength = np.rint(
                        255 * (local - local.min()) / max(local.max() - local.min(), 1e-30)
                    ).astype(np.uint8)
                    edges.extend(endpoints.tobytes())
                    weights.extend(strength.tobytes())
                tours.extend(np.stack((source, current, best)).astype("<u2").tobytes())
                metrics.append(
                    [float(trace[0]), float(trace[14]), float(trace[15]), float(trace[23])]
                )
            payload["data"][regime][behavior] = {
                "heat": base64.b64encode(heat).decode(),
                "edges": base64.b64encode(edges).decode(),
                "weights": base64.b64encode(weights).decode(),
                "tours": base64.b64encode(tours).decode(),
                "ranges": ranges,
                "metrics": metrics,
            }
    return payload


def render_dashboard(output: Path, report: Path) -> Path | None:
    payload = _pack_dashboard(output, report)
    if payload is None:
        return None
    template = r"""<!doctype html><html lang="zh"><head><meta charset="utf-8"><title>MMAS 历史路径强化机制</title>
<style>body{font-family:system-ui,sans-serif;margin:18px;color:#222}#controls{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:10px}label{font-size:14px}canvas{border:1px solid #ccc;background:white}.row{display:flex;gap:14px;flex-wrap:wrap}.card{border:1px solid #ddd;padding:10px;border-radius:5px}#caption{max-width:1050px}.legend span{margin-right:15px}.source{color:#d64b3c}.current{color:#2878b5}.global{color:#3d9140}</style></head>
<body><h1>MMAS 历史路径强化与 GP 源内重加权</h1><p id="caption">单个 baseline 中位实例的说明性重放。统计结论不来自此动画。</p>
<div id="controls"><label>来源策略 <select id="regime"></select></label><label>规则 <select id="behavior"></select></label><label>更新阶段 <select id="phase"><option value="0">更新前</option><option value="1">蒸发和下界保护后</option><option value="2">沉积后</option><option value="3">重启和可选上界处理后</option></select></label><button id="play">播放</button><label>轮次 <input id="iteration" type="range" min="0" max="200" value="0" step="1"></label><b id="iterationText"></b></div>
<div class="legend"><span>高信息素候选弧：黄→红</span><span class="source">实际强化路径（全部路径策略仅显示第一条）</span><span class="current">本轮最优路径</span><span class="global">历史全局最优路径</span></div>
<div class="row"><div class="card"><canvas id="spatial" width="620" height="620"></canvas></div><div class="card"><canvas id="heatmap" width="620" height="620"></canvas></div></div>
<p id="metrics"></p><h2>读图顺序</h2><ol><li>来源策略先决定哪些路径边可获得沉积。</li><li>GP 信息素树只在这些已选边内重分配固定预算。</li><li>信息素改变下一轮选择概率，进而改变构造路径、2-opt 输入和后续优良路径。</li><li>重复强化形成反馈。单轮变化很小也能在数百轮后产生差异。</li></ol>
<script>const P=__PAYLOAD__;
const sourceLabels={0:'本轮路径',1:'重启阶段最优路径',2:'全局最优路径'};
function bytes(s){const b=atob(s),a=new Uint8Array(b.length);for(let i=0;i<b.length;i++)a[i]=b.charCodeAt(i);return a}
function u16(s){const a=bytes(s);return new Uint16Array(a.buffer)}
const regime=document.getElementById('regime'),behavior=document.getElementById('behavior'),phase=document.getElementById('phase'),slider=document.getElementById('iteration');
Object.entries(P.regimes).forEach(([k,v])=>regime.add(new Option(v,k)));Object.entries(P.behaviors).forEach(([k,v])=>behavior.add(new Option(v,k)));
let timer=null;function path(ctx,tour,color,width,coords){ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();for(let i=0;i<tour.length;i++){const p=coords[tour[i]];const x=20+p[0]*580,y=600-p[1]*580;i?ctx.lineTo(x,y):ctx.moveTo(x,y)}ctx.stroke()}
function draw(){const d=P.data[regime.value][behavior.value],f=+slider.value,p=+phase.value,coords=P.coords;
 document.getElementById('iterationText').textContent=P.iterations[f];const ec=bytes(d.edges),ew=bytes(d.weights),tv=u16(d.tours),hc=bytes(d.heat);
 const s=document.getElementById('spatial').getContext('2d');s.clearRect(0,0,620,620);s.globalAlpha=.75;const edgeOffset=((f*4+p)*P.top)*4,weightOffset=(f*4+p)*P.top;
 for(let k=0;k<P.top;k++){const q=edgeOffset+k*4,u=ec[q]+256*ec[q+1],v=ec[q+2]+256*ec[q+3],w=ew[weightOffset+k];s.strokeStyle=`rgb(255,${Math.max(20,220-w)},20)`;s.lineWidth=.3+2*w/255;s.beginPath();s.moveTo(20+coords[u][0]*580,600-coords[u][1]*580);s.lineTo(20+coords[v][0]*580,600-coords[v][1]*580);s.stroke()}
 s.globalAlpha=.8;const tourOffset=f*3*501;path(s,tv.slice(tourOffset,tourOffset+501),'#d64b3c',1.6,coords);path(s,tv.slice(tourOffset+501,tourOffset+1002),'#2878b5',1.2,coords);path(s,tv.slice(tourOffset+1002,tourOffset+1503),'#3d9140',1.0,coords);
 const h=document.getElementById('heatmap').getContext('2d'),img=h.createImageData(20,20),offset=(f*4+p)*400;for(let i=0;i<400;i++){const v=hc[offset+i];img.data[4*i]=v;img.data[4*i+1]=45;img.data[4*i+2]=255-v;img.data[4*i+3]=255}const tmp=document.createElement('canvas');tmp.width=20;tmp.height=20;tmp.getContext('2d').putImageData(img,0,0);h.imageSmoothingEnabled=false;h.clearRect(0,0,620,620);h.drawImage(tmp,10,10,600,600);
 const m=d.metrics[f];document.getElementById('metrics').textContent=`强化来源=${sourceLabels[m[0]]??m[0]}；来源年龄=${m[1].toFixed(1)} 轮；停滞=${m[2].toFixed(1)} 轮；来源内沉积变异系数=${m[3].toFixed(3)}`}
[regime,behavior,phase,slider].forEach(x=>x.addEventListener('input',draw));document.getElementById('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;return}timer=setInterval(()=>{slider.value=(+slider.value+1)%P.iterations.length;draw()},180)};draw();</script></body></html>"""
    text = template.replace(
        "__PAYLOAD__", json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )
    path = report / "mechanism_dashboard.html"
    path.write_text(text)
    if path.stat().st_size > 50 * 1024**2:
        raise ValueError("交互 HTML 超过 50 MiB")
    return path


def _main_table(report: Path) -> list[dict]:
    rows = _read(report / "source_policy_performance.csv")
    selected = [
        row.copy()
        for row in rows
        if (
            (
                row["execution_configuration"].startswith("AS：")
                and row["expression_origin"].startswith("AS ")
            )
            or (
                row["execution_configuration"].startswith("MMAS：")
                and row["expression_origin"].startswith("MMAS ")
            )
        )
    ]
    for row in selected:
        for key in ("baseline_gap_percent", "gp_gap_percent", "gp_benefit_pp"):
            row[key] = float(row[key])
        for key in ("wins", "ties", "losses"):
            row[key] = int(row[key])
    return selected


def write_report(source: Path, output: Path, report: Path) -> Path:
    main = _main_table(report)
    old = source.parents[1] / "reports/mechanism-explanation-v1"
    effects = _read(old / "component_effects.csv")
    differences = _read(old / "component_effect_differences.csv")
    for row in effects:
        for key in ("移除后 GP 增益的变化（百分点）", "同时区间下限", "同时区间上限"):
            row[key] = float(row[key])
    for row in differences:
        for key in ("GP 增益变化之差（百分点）", "同时区间下限", "同时区间上限"):
            row[key] = float(row[key])
    quality = _read(report / "continuation_quality_across_snapshots_summary.csv")
    source_support = _read(report / "same_state_source_support_summary.csv")
    trajectory = _read(report / "pheromone_trajectory_summary.csv")
    expressions = _read(report / "expression_behavior.csv")
    for row in expressions:
        row["aggregate_global_history_source_edge_change"] = float(
            row["aggregate_global_history_source_edge_change"]
        )
    replay = read_json(report / "replay_analysis.json")

    def q(metric, elapsed=500):
        selected = [
            row
            for row in quality
            if row["metric"] == metric
            and row["state_owner"] == "不使用 GP 产生的状态"
            and int(row["continuation_iterations"]) == elapsed
        ]
        return float(selected[0]["mean"]) if selected else np.nan

    source_overlap = sorted(
        {
            (
                row["state_owner"],
                int(row["snapshot_iteration"]),
                float(row["mean"]),
            )
            for row in source_support
            if row["metric"] == "source_overlap"
            and row["historical_source"] == "历史全局最优路径"
        }
    )
    mean_overlap = np.mean([row[2] for row in source_overlap]) if source_overlap else np.nan
    mean_nonshared = 500 * (1 - mean_overlap)
    max_nonshared = max((500 * (1 - row[2]) for row in source_overlap), default=np.nan)

    trajectory_index = {
        (row["behavior"], int(row["iteration"]), row["metric"]): float(row["mean"])
        for row in trajectory
    }
    keyframes = []
    for iteration in (100, 500, 2500, 5000):
        for behavior in ("不使用 GP", "三个固定 GP 表达式的均值"):
            keyframes.append(
                {
                    "iteration": iteration,
                    "behavior": behavior,
                    "candidate_effective_arcs": round(
                        trajectory_index[(behavior, iteration, "candidate_effective_arcs")], 2
                    ),
                    "source_edge_lift": round(
                        trajectory_index[(behavior, iteration, "source_edge_lift")], 3
                    ),
                    "reference_edge_lift": round(
                        trajectory_index[(behavior, iteration, "reference_edge_lift")], 3
                    ),
                    "post_unique_tours": round(
                        trajectory_index[(behavior, iteration, "post_unique_tours")], 2
                    ),
                }
            )
    lines = [
        "# MMAS 与 2-opt 中历史路径强化为何压缩 GP 的边际收益",
        "",
        "## 直接回答",
        "",
        "当前开发集和同状态实验指向同一个主因：不是重启，也不是信息素下界保护。主因是强化来源的选择。完整 MMAS 会按日程反复选择历史优良路径。这个步骤先决定沉积可以落在哪些边上。GP 信息素树随后只能在这些已选边内部重分配固定预算。它不能把预算移到来源路径之外。",
        "",
        "当框架从“强化全部本轮路径”改为“只强化一条本轮最优路径”时，baseline 已显著增强，GP 的边际空间同时缩小。再改为历史优良路径强化后，baseline 进一步增强，GP 增益接近零或变为负值。这个模式在 AS 和 MMAS 内部都出现。因此，结果不能只归因于 MMAS 这个名称。它来自优良路径来源选择和重复反馈。",
        "",
        "早期八格控制实验中，关闭历史路径强化使 GP 增益增加 0.38566 个百分点，95% 同时区间为 [0.10470, 0.66662]。关闭重启的变化为 -0.00974 [-0.04497, 0.02549]。关闭信息素下界保护的变化为 -0.01404 [-0.04890, 0.02081]。历史路径强化的移除效应显著大于另外两项。该结果来自开发集，不是尚未运行的独立确认集。",
        "",
        "## 作用链",
        "",
        r"设本轮被框架选中的来源边集为 $S_t$。不使用 GP 时，来源内部使用均匀权重。使用 GP 时，每条来源边的权重为",
        "",
        r"\[w_e=1+\frac{1}{3}\tanh(g(e)),\qquad e\in S_t.\]",
        "",
        r"固定总预算 $B_t$ 后，实际沉积为",
        "",
        r"\[D_t(e)=B_t\frac{w_e}{\sum_{j\in S_t}w_j},\qquad e\in S_t.\]",
        "",
        r"对 $e\notin S_t$，$D_t(e)=0$。因此，框架先决定支持集 $S_t$。GP 只决定支持集内的比例。完整 MMAS 反复选择重启阶段最优或全局最优路径。相同优良边被多次强化。信息素集中后，这些边更可能进入下一轮路径。2-opt 又把较好的构造路径映射到局部最优解。新的优良解随后进入历史记忆。这个循环构成正反馈。",
        "",
        "单次历史来源替换的全矩阵平均变化很小，因为 TSP500 有约 25 万条有向弧，而两条优良路径通常共享大部分边。当前同状态结果中，历史路径与本轮最优路径的平均边重合率约为 "
        + (f"{mean_overlap:.4f}" if source_overlap else "—")
        + f"。每条路径平均约有 {mean_nonshared:.1f} 条边不被另一条路径共享；早期差异最大的快照约有 {max_nonshared:.1f} 条。"
        + "变化集中在这些边和包含这些边的构造上下文中。全矩阵相对 L1 会稀释这种局部作用。报告因此同时给出来源边局部变化和受影响上下文的概率 TV。",
        "",
        "同状态起点下，1 轮后的原生来源与仅本轮最优来源几乎没有质量差异。续跑 500 轮后，移除历史来源使 GP 增益增加 "
        + f"{q('移除历史来源后 GP 增益变化（百分点）'):.5f} 个百分点。"
        + "这说明主要作用不是一次大跳变，而是来源选择、信息素和后续路径之间的累积反馈。",
        "",
        "## 最终质量来源对照",
        "",
        "GP 增益定义为同条件不使用 GP 的 gap 减使用 GP 的 gap。正值表示 GP 更好。每个实例先平均 5 个 ACO 种子和三个固定表达式。表中是 32 个开发实例。",
        "",
        markdown_table(
            main,
            [
                ("execution_configuration", "执行配置"),
                ("expression_origin", "表达式来源"),
                ("baseline_gap_percent", "不使用 GP 的 gap（%）"),
                ("gp_gap_percent", "使用 GP 的 gap（%）"),
                ("gp_benefit_pp", "GP 改进（百分点）"),
                ("wins", "胜"),
                ("ties", "平"),
                ("losses", "负"),
            ],
        ),
        "",
        "![来源策略与质量](source_policy_performance.png)",
        "",
        "## 哪个 MMAS 组件造成主要差异",
        "",
        markdown_table(
            effects,
            [
                ("移除的组件", "移除的组件"),
                ("移除后 GP 增益的变化（百分点）", "GP 增益变化（百分点）"),
                ("同时区间下限", "同时区间下限"),
                ("同时区间上限", "同时区间上限"),
            ],
        ),
        "",
        markdown_table(
            differences,
            [
                ("直接比较", "直接比较"),
                ("GP 增益变化之差（百分点）", "效应差（百分点）"),
                ("同时区间下限", "同时区间下限"),
                ("同时区间上限", "同时区间上限"),
            ],
        ),
        "",
        "这组控制实验支持“历史路径强化是主要组件”的判断。它不支持“重启是主因”或“下界保护是主因”。重启和下界仍可能改变绝对质量与长期状态，但当前固定表达式的边际收益差主要由来源策略解释。",
        "",
        "## 同状态即时干预",
        "",
        "![同状态信息素传递](same_state_tau_transfer.png)",
        "",
        "左图将边分为历史来源特有、本轮最优特有、共有和其他候选弧。中图只在候选集中含来源差异边时计算主要概率响应。右图比较同一来源、同一预算下关闭和启用 GP 更新树。它们分别回答两个问题：框架选择哪条路径，以及 GP 如何在该路径内分配预算。",
        "",
        "## 长期信息素和群体行为",
        "",
        "![信息素集中轨迹](pheromone_concentration.png)",
        "",
        "![历史强化事件](historical_reinforcement_events.png)",
        "",
        markdown_table(
            keyframes,
            [
                ("iteration", "轮次"),
                ("behavior", "规则"),
                ("candidate_effective_arcs", "有效候选弧数"),
                ("source_edge_lift", "来源边提升倍数"),
                ("reference_edge_lift", "参考路径边提升倍数"),
                ("post_unique_tours", "2-opt 后不同路径数"),
            ],
        ),
        "",
        "第 100 轮，GP 条件的来源边提升更高，有效候选弧更少，2-opt 后不同路径也更少。第 500 轮及以后，该方向会随重启和状态反馈改变。GP 不是单向加强信息素集中。它改变的是各阶段的集中—释放轨迹。自然轨迹可以说明这种时变行为，但不能单独识别因果效应。因果方向主要由同状态干预提供。",
        "",
        "## 2-opt 是否抹去 GP 的作用",
        "",
        "![同状态续跑与 2-opt](same_state_response.png)",
        "",
        "2-opt 会减少构造路径之间的边集差异。它没有在所有状态和时域中完全消除差异。最终最好解是极值统计。平均构造质量或平均 2-opt 后质量改善，不保证最终极值一定改善。完整 MMAS baseline 已很强时，这个差别更明显。",
        "",
        "## 三个固定表达式",
        "",
        markdown_table(
            expressions,
            [
                ("seed", "训练种子"),
                ("transition_tree_active", "构造决策树非零"),
                ("pheromone_tree_active", "信息素更新树非零"),
                ("selected_for_animation", "用于动画"),
                (
                    "aggregate_global_history_source_edge_change",
                    "历史来源边变化 / 2B",
                ),
                ("selection_passed_noninferiority", "通过非劣门槛"),
                ("final_deployed", "最终部署"),
            ],
        ),
        "",
        "![表达式行为](expression_behavior.png)",
        "",
        "动画固定使用种子 81002，因为它是三个 MMAS 冠军中唯一两棵树都非零的表达式。这个选择只服务于可视化。静态汇总保留三个表达式。三个原始冠军均未通过训练时的非劣选择门槛，也未成为最终部署策略。因此，这里分析的是固定原始表达式的机制，不是已部署策略。",
        "",
        "## 交互可视化",
        "",
    ]
    if replay.get("status") == "complete":
        lines += [
            "[打开离线交互页面](mechanism_dashboard.html)。页面可以切换四种来源策略、baseline/GP、轮次和四个更新阶段。空间图显示高信息素候选弧及实际来源、本轮最优和历史全局最优路径。热图按参考路径顺序重排。",
            "",
        ]
    else:
        lines += [
            "定向单实例重放尚未全部完成。静态控制实验图已生成。重放完成后运行 `render` 会加入离线交互页面。",
            "",
        ]
    lines += [
        "## 结论边界",
        "",
        "现有证据可以支持以下结论：完整 MMAS 中历史优良路径的来源选择和重复强化，显著压缩了固定 GP 规则的边际改进空间。GP 更新树的作用域受来源路径限制。单轮局部变化通过后续选择和局部搜索反馈累积。",
        "",
        "现有证据不能证明该机制对所有实例分布、重新训练后的规则或未知 GP 表达式都成立。280 项独立确认仍因共享盘空间不足暂停。报告没有使用其不完整子集。动画是说明性个案，不是统计证明。",
        "",
        "## 产物",
        "",
        "- `source_policy_performance.csv`：来源策略最终质量。",
        "- `same_state_source_support_per_instance.csv`：来源支持集的即时信息素变化。",
        "- `same_state_probability_per_instance.csv`：受影响和未受影响构造上下文。",
        "- `same_state_update_tree_per_instance.csv`：同来源同预算下的 GP 更新树作用。",
        "- `pheromone_trajectory_per_instance.csv`：信息素集中与路径对齐轨迹。",
        "- `continuation_quality_per_instance.csv`：同状态 1、25、100、500 轮续跑。",
        "- `two_opt_survival_per_instance.csv`：2-opt 前后结构差异。",
        "- `verification.json`：完整性和范围核验。",
        "",
    ]
    path = report / "report_zh.md"
    path.write_text("\n".join(lines))
    return path


def render(source: Path, output: Path, report: Path) -> Path:
    if read_json(report / "analysis_manifest.json", {}).get("status") != "analysis_complete":
        raise ValueError("先完成 analyze")
    plot_source_policy(report)
    plot_same_state_transfer(report)
    plot_pheromone_trajectory(report)
    plot_response_and_local_search(report)
    plot_historical_events(report)
    plot_expression_behavior(report)
    dashboard = render_dashboard(output, report)
    write_report(source, output, report)
    files = {}
    for path in sorted(report.iterdir()):
        if path.is_file() and path.name not in ("report_manifest.json", "verification.json"):
            files[path.name] = file_hash(path)
    atomic_json(
        report / "report_manifest.json",
        {
            "status": "complete",
            "generated_at": now(),
            "files": files,
            "dashboard": dashboard.name if dashboard else None,
            "scope": "开发集固定表达式机制报告；独立确认暂停；动画不作为统计证据",
        },
    )
    return report / "report_zh.md"
