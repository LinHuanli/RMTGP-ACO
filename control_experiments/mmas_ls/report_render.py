"""从同一份统计结果绘图并渲染中文报告，不在文稿中手填实验数值。"""
from __future__ import annotations
from pathlib import Path
import numpy as np

from .statistics import CONDITIONS,write_csv


def pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False,
                         "pdf.fonttype":42,"savefig.dpi":160})
    return plt


def save(plt,fig,path):
    fig.savefig(path.with_suffix(".pdf"))
    fig.savefig(path.with_suffix(".png"),dpi=160)
    plt.close(fig)


def interval_band(values):
    """32 个实例的点态带，仅用于曲线描述；输入已平均种子与固定冠军。"""
    rng=np.random.default_rng(2026091702)
    draws=rng.integers(0,len(values),size=(10000,len(values)))
    # 分块避免同时物化 [10000,32,201] 的大数组。
    means=np.concatenate([values[x].mean(1) for x in np.array_split(draws,20)],axis=0)
    return np.quantile(means,[.025,.975],axis=0)


def plot_quality(target,arrays,tables):
    from .research_report import CURVE_ITERATIONS,VARIANTS,MODES
    plt=pyplot()
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout="constrained")
    for ax,variant in zip(axes,VARIANTS):
        for s,mode in enumerate(MODES):
            rows=[r for r in tables["main_results"] if r["variant"]==variant and r["mode"]==mode]
            ax.scatter(np.arange(5)+(s-1)*.16,[r["mean_gap"] for r in rows],s=28,label=mode)
        ax.set(xticks=np.arange(5),xticklabels=["Baseline","81001","81002","81003","GP mean"],
               ylabel="Reference gap (%)",title=variant.upper()+" + 2-opt; 32 instances x 5 seeds")
        ax.legend(fontsize=8)
    save(plt,fig,target/"main_results")
    rows=[r for r in tables["numerical_effects"] if r["mode"]!="legacy" and r["model"] not in ("baseline","GP_mean")]
    fig,ax=plt.subplots(figsize=(9,6),layout="constrained")
    m=np.array([r["change_vs_legacy_pp"] for r in rows]);lo=np.array([r["ci_low_pp"] for r in rows]);hi=np.array([r["ci_high_pp"] for r in rows])
    ax.errorbar(m,np.arange(len(rows)),xerr=[np.maximum(0,m-lo),np.maximum(0,hi-m)],fmt="o",capsize=2)
    ax.axvspan(-.01,.01,color="gray",alpha=.15);ax.axvline(0,color="black",lw=.7)
    ax.set(yticks=np.arange(len(rows)),yticklabels=[f"{r['variant'].upper()} {r['model']} {r['mode']}" for r in rows],
           xlabel="Stable mode - legacy (pp); pointwise 95% CI",title="Fixed champions; no retraining; descriptive intervals")
    ax.invert_yaxis();save(plt,fig,target/"numeric_effects")
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),layout="constrained")
    f=tables["factorial"];pos=np.arange(8)
    axes[0].scatter(pos-.07,[r["baseline_gap"] for r in f],label="Baseline")
    axes[0].scatter(pos+.07,[r["mean_gap"] for r in f],label="GP mean",marker="s")
    axes[0].set(xticks=pos,xticklabels=CONDITIONS,ylabel="Reference gap (%)",title="Absolute quality");axes[0].legend()
    m=np.array([r["delta_pp"] for r in f]);lo=np.array([r["ci_low_pp"] for r in f]);hi=np.array([r["ci_high_pp"] for r in f])
    axes[1].errorbar(pos,m,yerr=[m-lo,hi-m],fmt="o",capsize=3)
    axes[1].axhspan(-.01,.01,color="gray",alpha=.15);axes[1].axhline(0,color="black",lw=.7)
    axes[1].set(xticks=pos,xticklabels=CONDITIONS,ylabel="GP - baseline (pp)",title="95% simultaneous CI, 18-contrast family")
    save(plt,fig,target/"factorial_quality")
    fig,ax=plt.subplots(figsize=(9,4),layout="constrained")
    ax.bar(pos-.18,[r["baseline_change_pp"] for r in f],width=.36,label="Baseline change")
    ax.bar(pos+.18,[r["gp_change_pp"] for r in f],width=.36,label="GP change")
    ax.axhline(0,color="black",lw=.7);ax.legend()
    ax.set(xticks=pos,xticklabels=CONDITIONS,ylabel="Gap change from C111 (pp)",title="Positive change means worse absolute quality")
    save(plt,fig,target/"absolute_changes")
    rows=tables["contrasts"]
    fig,ax=plt.subplots(figsize=(9,7),layout="constrained")
    m=np.array([r["mean_pp"] for r in rows]);lo=np.array([r["ci_low_pp"] for r in rows]);hi=np.array([r["ci_high_pp"] for r in rows])
    ax.errorbar(m,np.arange(18),xerr=[m-lo,hi-m],fmt="o",capsize=2)
    ax.axvline(0,color="black",lw=.7);ax.axvspan(-.01,.01,color="gray",alpha=.15)
    ax.set(yticks=np.arange(18),yticklabels=[r["contrast"] for r in rows],xlabel="Paired effect (pp); simultaneous 95% CI",
           title="Historical implementation; fixed champions; development set")
    ax.invert_yaxis();save(plt,fig,target/"factorial_effects")
    fig,axes=plt.subplots(1,3,figsize=(12,4),layout="constrained")
    values={r["condition"]:r["delta_pp"] for r in f}
    for axis,(ax,factor) in enumerate(zip(axes,"RFH")):
        others=[i for i in range(3) if i!=axis]
        for a in (0,1):
            for b in (0,1):
                bits=[0]*3;bits[others[0]]=a;bits[others[1]]=b
                y=[]
                for state in (0,1):
                    bits[axis]=state;y.append(values["C"+"".join(map(str,bits))])
                ax.plot([0,1],y,"o-",label=f"{'RFH'[others[0]]}={a}, {'RFH'[others[1]]}={b}")
        ax.axhline(0,color="black",lw=.7);ax.set(xticks=[0,1],xlabel=factor,ylabel="GP - baseline (pp)",title="Conditional means (descriptive)");ax.legend(fontsize=7)
    save(plt,fig,target/"interactions")
    curve_rows=[]
    fig,axes=plt.subplots(2,4,figsize=(13,6.5),layout="constrained",sharex=True)
    for c,ax in enumerate(axes.flat):
        for label,v in (("Baseline",arrays["f_curve"][c,0].mean(1)),("GP mean",arrays["f_curve"][c,1:].mean((0,2)))):
            mean=v.mean(0);lo,hi=interval_band(v)
            ax.plot(CURVE_ITERATIONS,mean,label=label);ax.fill_between(CURVE_ITERATIONS,lo,hi,alpha=.13)
            curve_rows.extend({"cohort":"P1","condition":CONDITIONS[c],"model":label,"iteration":int(it),"mean_gap":float(m),"point_low":float(l),"point_high":float(h)}
                              for it,m,l,h in zip(CURVE_ITERATIONS,mean,lo,hi))
        ax.set(title=CONDITIONS[c],xlabel="ACO iteration",ylabel="Incumbent gap (%)",xscale="log")
    axes.flat[0].legend();save(plt,fig,target/"factorial_anytime")
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout="constrained")
    for v,ax in enumerate(axes):
        for label,x in (("Baseline",arrays["n_curve"][v,0,0].mean(1)),("GP mean",arrays["n_curve"][v,0,1:].mean((0,2)))):
            mean=x.mean(0);lo,hi=interval_band(x)
            ax.plot(CURVE_ITERATIONS,mean,label=label);ax.fill_between(CURVE_ITERATIONS,lo,hi,alpha=.15)
            curve_rows.extend({"cohort":"N1_"+VARIANTS[v],"condition":"C111","model":label,"iteration":int(it),"mean_gap":float(m),"point_low":float(l),"point_high":float(h)}
                              for it,m,l,h in zip(CURVE_ITERATIONS,mean,lo,hi))
        ax.set(title=VARIANTS[v].upper()+" + 2-opt, legacy",xlabel="ACO iteration",ylabel="Incumbent gap (%)",xscale="log");ax.legend()
    save(plt,fig,target/"native_anytime");write_csv(target/"anytime_curves.csv",curve_rows)


def plot_mechanisms(target,arrays):
    from .report_diagnostics import WINDOW_FIELDS,SAMPLE_FIELDS
    plt=pyplot();w=arrays["windows"];s=arrays["samples"]
    panels=((w,WINDOW_FIELDS,"ph_saturation","PH saturation (all source edges / 25 iterations)"),
            (s,SAMPLE_FIELDS,"deposit_ad_same_source_budget","Deposit L1 / same-source budget (endpoints)"),
            (w,WINDOW_FIELDS,"source_age","Source age (iterations)"))
    for values,names,field,title in panels:
        fig,axes=plt.subplots(2,4,figsize=(13,6),layout="constrained",sharex=True)
        t=np.arange(25,5001,25) if values.shape[-2]==200 else np.array([1]+list(range(25,5001,25)))
        for c,ax in enumerate(axes.flat):
            k=names.index(field)
            ax.plot(t,values[c,0,...,k].mean((0,1)),label="Baseline",lw=1)
            ax.plot(t,values[c,1:,...,k].mean((0,1,2)),label="GP mean",lw=1)
            ax.set(title=CONDITIONS[c],xlabel="ACO iteration",ylabel=field)
        axes.flat[0].legend();fig.suptitle(title+"; descriptive trajectories")
        save(plt,fig,target/("mechanism_"+field))
    fig,axes=plt.subplots(2,3,figsize=(12,7),layout="constrained")
    overview=((w,WINDOW_FIELDS,"ls_gain_pp"),(w,WINDOW_FIELDS,"ls_retention"),
              (s,SAMPLE_FIELDS,"tr_entropy_probe"),(w,WINDOW_FIELDS,"source_repeat_duration_end"),
              (w,WINDOW_FIELDS,"floor_fraction"),(w,WINDOW_FIELDS,"ph_tanh_std"))
    for ax,(values,names,field) in zip(axes.flat,overview):
        t=np.arange(25,5001,25) if values.shape[-2]==200 else np.array([1]+list(range(25,5001,25)))
        for c,condition in enumerate(CONDITIONS):ax.plot(t,values[c,1:,...,names.index(field)].mean((0,1,2)),label=condition,lw=1)
        ax.set(title=field,xlabel="ACO iteration",ylabel="Fixed GP champion mean")
    axes.flat[0].legend(ncol=2,fontsize=7);save(plt,fig,target/"mechanism_overview")
    # 每位冠军单列，防止总均值掩盖不同表达式的行为。
    fig,axes=plt.subplots(2,4,figsize=(13,6),layout="constrained",sharex=True)
    for c,ax in enumerate(axes.flat):
        for p in range(3):ax.plot(np.arange(25,5001,25),w[c,p+1,...,WINDOW_FIELDS.index("ph_saturation")].mean((0,1)),label=str(81001+p),lw=1)
        ax.set(title=CONDITIONS[c],xlabel="ACO iteration",ylabel="PH saturation")
    axes.flat[0].legend();save(plt,fig,target/"champion_ph_saturation")


def markdown_table(rows,columns):
    """所有正文数字来自同一个统计对象。None 显式写为未定义。"""
    def value(x):
        if x is None:return "—"
        if isinstance(x,(float,np.floating)):return f"{x:.5f}"
        return str(x).replace("|","\\|")
    lines=["| "+" | ".join(title for key,title in columns)+" |","| "+" | ".join("---" for _ in columns)+" |"]
    lines.extend("| "+" | ".join(value(row[key]) for key,title in columns)+" |" for row in rows)
    return "\n".join(lines)


def render(target,tables,meta,mechanisms,event_summary,provenance):
    """中文研究报告。观察、解释及尚缺实验分开书写。"""
    main=[r for r in tables["main_results"] if r["mode"]=="legacy"]
    numeric=[r for r in tables["numerical_effects"] if r["model"]=="GP_mean"]
    factor=tables["factorial"];effects=tables["contrasts"]
    h=next(r for r in effects if r["contrast"]=="E_H")
    c111=factor[0];c110=factor[1]
    gp_mechanisms=[r for r in mechanisms if r["model"]=="GP_mean"]
    validation=meta["validation"]
    native_gp=next(r for r in mechanisms if r["condition"]=="C111" and r["model"]=="GP_mean")
    no_history_gp=next(r for r in mechanisms if r["condition"]=="C110" and r["model"]=="GP_mean")
    native_base=next(r for r in mechanisms if r["condition"]=="C111" and r["model"]=="baseline")
    no_history_base=next(r for r in mechanisms if r["condition"]=="C110" and r["model"]=="baseline")
    h_process=[{"metric":label,"native_base":native_base[key],"no_h_base":no_history_base[key],
                "native_gp":native_gp[key],"no_h_gp":no_history_gp[key]} for key,label in (
                    ("pre_gap","构造路径平均 gap (%)"),("post_gap","2-opt 后路径平均 gap (%)"),
                    ("ls_retention","构造边保留率"),("tr_entropy_probe","TR 探针熵"),
                    ("restarts_per_solve","每求解重启数"))]
    lines=["# MMAS–2-opt 中固定 GP 策略的收益与机制：开发集受控实验报告", "",
           f"生成时间：{provenance['generated_at']}。实验批次：`numerical-v1`。状态：质量结果及完整诊断文件核验通过。", "",
           "## 摘要", "",
           "本报告分析 32 个 Uniform TSP500 实例。每个实例运行 5 个 ACO seeds。每个框架固定三个已训练冠军。所有质量实验使用 32 只蚂蚁、5000 轮和 ACOTSP 风格 2-opt。这里没有新训练，也没有使用确认集。", "",
           "AS 的固定 GP 冠军明显降低了本开发集上的平均 reference gap。MMAS 的原始冠军均值略差于其 baseline。中心化数值修正通过了独立输入核验，但旧冠军的平均解质量没有因此改善。这个结果不回答修正后重新训练能否提高质量。", "",
           f"在历史实现中，移除历史路径强化来源 H 后，GP 相对 baseline 的净差改变 {h['mean_pp']:.5f} 个百分点，95% 同时区间为 [{h['ci_low_pp']:.5f}, {h['ci_high_pp']:.5f}]。"
           f"但 baseline 的 gap 增加 {c110['baseline_change_pp']:.5f}，GP 的 gap 也增加 {c110['gp_change_pp']:.5f}。因此，相对优势主要扩大于 baseline 退化更多，而不是 GP 绝对质量提高。", "",
           "本结果只支持开发集、固定冠军和历史实现下的干预效应。它不能证明历史强化与 GP 功能重叠，也不能证明 GP 可以替代局部搜索。", "",
           "## 1. 动机与实验对象", "",
           "研究问题是：为什么同一类可学习残差在 AS+2-opt 中有较大收益，在 MMAS+2-opt 中收益有限？本轮将数值环境与 MMAS 组件分开干预。前者检查 terminal 计算的影响。后者估计重启、信息素下界和历史路径来源对 GP 净收益的影响。", "",
           "### 1.1 固定配置", "",
           markdown_table([
               {"k":"数据","v":"32 个新 Uniform TSP500 开发实例；5 个共同 ACO seeds；配对单位为实例"},
               {"k":"模型","v":"每框架固定 raw 冠军 81001、81002、81003；不在本数据上重新选择"},
               {"k":"ACO","v":"32 ants；5000 iterations；alpha=1，beta=2；AS rho=0.5，MMAS rho=0.2"},
               {"k":"候选与局部搜索","v":"construction candidate=20；LS candidate=20；ACOTSP 2-opt + DLB；每轮全部蚂蚁"},
               {"k":"GP 集成","v":"TR residual；PH budget_residual；两者 gamma=1/3；来源预算匹配"},
               {"k":"数值后端","v":"cuda_tiled_v2；fp32_fast 搜索；最终路径长度用 CPU FP64 重算"},
               {"k":"数值对照","v":"legacy / centered_fp32 / centered_fp64；后者仅统计与归一化使用 FP64"},
               {"k":"本轮任务","v":"18 个型号/数值验收 + 40 个三模式配对任务 + 40 个历史八格任务 = 98"},
           ],[("k","项目"),("v","设置")]),"",
           "数值对照含 3840 个逻辑求解。八格实验含 5120 个逻辑求解。两类实验有共同配置，不能将其简单相加作为独立样本量。主分析每项只有 32 个配对实例。", "",
           "### 1.2 R/F/H 的定义", "",
           markdown_table([
               {"k":"R","one":"原生重启写操作","zero":"关闭重启；不删除全部历史信息"},
               {"k":"F","one":"候选有向弧蒸发后的物理信息素下界","zero":"取消物理 floor；保留标称范围与 TauHeadroom 定义"},
               {"k":"H","one":"原生 IB/RB/GB 路径来源日程","zero":"实际强化只用 iteration-best 边；仍按原生影子来源匹配总预算"},
           ],[("k","开关"),("one","1"),("zero","0")]),"",
           "条件名 C111 的三个位置依次是 R、F、H。原生 MMAS–LS 分支没有硬 tau_max 上界裁剪。H=0 不是完全无记忆算法。实际来源元数据与影子预算来源分开处理。AS 的 native 来源为 32 只当前蚂蚁，不是 32 条 iteration-best 路径。", "",
           "## 2. 指标、统计单位与不确定性", "",
           r"\[g=100(L/L_{\mathrm{ref}}-1),\qquad \Delta=g_{GP}-g_{baseline}.\]", "",
           "参考长度来自可行参考路径，在连续欧氏距离下用 FP64 重算。它不是已证明的最优值。gap 越小越好，负值保留。Δ 以百分点（pp）计，负值表示 GP 更好。", "",
           "先在每个实例内平均 5 个 ACO seeds。三个固定冠军分别列出，整体结果再对冠军等权平均。baseline 只保留一份。重采样时整块抽取实例，同时保留所有条件及模型的配对关系。", "",
           "八格主比较包含三个移除效应、三个效应大小差、三个二阶交互、一个三阶交互和八个 Δ，共 18 项。使用 30,000 次 bootstrap max-t；seed=2026091701；报告 95% 同时区间。每冠军附表分别校正自己的 18 项，不声称三位冠军合起来具有整体 95% 覆盖率。", "",
           "数值对照、主结果描述及曲线采用 10,000 次实例 bootstrap 的点态区间，seed=2026091702。它们不是多重比较校正后的区间。实际意义阈值为 0.01 pp。区间不含零不等于整个区间超过该阈值；不显著也不等于无作用。", "",
           "AUC 是 5000 个 incumbent gap 的等权平均，不是另一次短预算运行。曲线长度来自 GPU FP32 incumbent，最终表中路径长度经过 CPU FP64 重算，因此末点允许微小舍入差。胜/平/负先对实例内 seeds 求平均，再以 ±0.01 pp 分类。最差 10% 指按 Δ 排序最大的 4 个实例均值。", "",
           "## 3. RQ1：固定冠军相对 baseline 的表现", "",
           markdown_table(main,[("variant","框架"),("model","模型"),("mean_gap","gap (%)"),("delta_pp","Δ (pp)"),
                               ("ci_low_pp","点态下限"),("ci_high_pp","点态上限"),("mean_auc","平均 incumbent gap"),("wins","胜"),("ties","平"),("losses","负")]),"",
           "![主结果](main_results.png)","",
           "图 1：不同数值模式的最终 gap。冠军是离散模型，不用连接线暗示连续关系。", "",
           "AS 与 MMAS 主表是各自原生参数下的结果比较，不是仅改变一个组件的实验。两者的信息素更新、来源数量、rho 等不同，不能把全部质量差都归因于某一个 MMAS 机制。", "",
           "![原配置曲线](native_anytime.png)","",
           "图 2：legacy 原配置的 incumbent 曲线。横轴为同一次 5000 轮运行的迭代数。阴影为实例级点态 95% 区间，不表示逐轮显著性。", "",
           f"MMAS 原配置还存在最终质量与全过程平均值的差别：GP 的平均 incumbent gap 为 {c111['gp_auc']:.5f}%，"
           f"baseline 为 {c111['baseline_auc']:.5f}%。GP 的该均值较低，但最终 gap 较高。不能把最终劣势写成全过程都更差。"
           "这个均值按迭代数计算，不是按实际耗时计算，因此也不能直接宣称等时间预算下更好。", "",
           "三个 MMAS raw 冠军在历史训练选择中的部署 gate 均未通过，因此主表不能称为实际回退部署策略的表现。下表单列冻结的 legacy 部署决策。新数值模式没有重新验收部署 gate。", "",
           markdown_table(tables["legacy_deployment"],[("variant","框架"),("deployment","历史部署策略"),("gap","gap (%)")]),"",
           "## 4. RQ2：数值修正是否改善旧冠军", "",
           "legacy 用 FP32 二阶原点矩计算方差。接近常量的输入可能出现消去误差。中心化实现先平移输入，再计算均值与中心化平方差；接近的信息素使用稳定的 log 比值。修正没有删除 terminal，没有增大保护 epsilon，也没有改动残差范围。", "",
           markdown_table(numeric,[("variant","框架"),("mode","数值模式"),("mean_gap","GP 均值 gap (%)"),
                                  ("change_vs_legacy_pp","相对 legacy (pp)"),("ci_low_pp","点态下限"),("ci_high_pp","点态上限")]),"",
           "![数值配对差](numeric_effects.png)","",
           "图 3：每位固定冠军的配对变化。三种模式的 baseline 最终 tour、anytime、长度和 gap 已核验一致。所有冠军的模式对照见 [完整数值表](numerical_effects.csv)。", "",
           "三个 GPU 型号上的稳定模式均通过已保存输入的 FP64 oracle。legacy 通过的是历史轨迹等价检查，并不表示其 terminal 数学值正确。元素级失败次数不是独立样本数，详见 [逐 terminal 核验表](terminal_oracle.csv)。", "",
           "观察上，修正后的冠军均值没有改善。点态区间仍包含零，不能声称稳定模式导致统计显著的质量下降。旧冠军是在旧数值环境中选择的；本轮不能判断稳定环境中重新训练的上限。", "",
           "## 5. RQ3：R/F/H 如何影响 GP 净收益", "",
           "### 5.1 全部八格与绝对变化", "",
           markdown_table(factor,[("condition","条件"),("baseline_gap","baseline gap"),("mean_gap","GP gap"),("delta_pp","Δ (pp)"),
                                 ("ci_low_pp","同时下限"),("ci_high_pp","同时上限"),("baseline_change_pp","baseline 相对 C111"),("gp_change_pp","GP 相对 C111")]),"",
           "![八格质量](factorial_quality.png)","",
           "![绝对变化](absolute_changes.png)","",
           "图 4–5：同条件相对差和相对 C111 的绝对变化。绝对变化为正表示退化。不能将削弱 baseline 后的相对优势解释成改进算法。", "",
           f"C111 的 GP 净差为 {c111['delta_pp']:.5f} pp，同时区间为 [{c111['ci_low_pp']:.5f}, {c111['ci_high_pp']:.5f}]。区间不含零，但跨越 0.01 pp 阈值，因此不能说已经确认超过预设实际意义阈值的劣化。", "",
           f"移除 H 的 C110 中，GP gap 为 {c110['mean_gap']:.5f}%，仍差于完整 baseline 的 {c111['baseline_gap']:.5f}%。H 对 baseline 的绝对收益大于对当前 GP 的收益。这个差异可以解释相对优势变化，但不能直接证明功能重叠。", "",
           "### 5.2 条件效应与交互", "",
           r"\[E_H=\Delta_{110}-\Delta_{111},\quad E_R=\Delta_{011}-\Delta_{111},\quad E_F=\Delta_{101}-\Delta_{111}.\]", "",
           "这些是原配置附近的条件移除效应，不是跨所有背景平均的普遍因素排名。二阶项采用差上差，三阶项比较两种 H 背景下的 R×F 差上差。精确符号由 [统计代码](../../statistics.py) 固定。", "",
           markdown_table(effects,[("contrast","对比"),("mean_pp","均值 (pp)"),("ci_low_pp","同时下限"),("ci_high_pp","同时上限"),("practical_class","实际意义判断")]),"",
           "![同时区间](factorial_effects.png)","",
           "![条件交互](interactions.png)","",
           "图 6–7：18 项同时区间和条件均值。H 的移除效应及其与 R、F 的交互需要共同解释。R 或 F 的单项区间跨零不能用于排除它们的重要性。", "",
           "![八格质量轨迹](factorial_anytime.png)","",
           "图 8：八格 incumbent 曲线。全部条件都展示；没有根据结果挑选条件。每冠军的 18 项区间见 [附表](per_champion.csv)。", "",
           "### 5.3 实例异质性与尾部表现", "",
           markdown_table(factor,[("condition","条件"),("wins","胜"),("ties","平"),("losses","负"),
                                 ("worst10_delta_pp","最差4实例 Δ (pp)"),("baseline_auc","baseline AUC"),("gp_auc","GP AUC")]),"",
           "例如 C100 的平均 Δ 为负，但负例数多于胜例数。均值优势不能解释成大多数实例都改善。尾部值描述风险，不是新增的显著性检验。", "",
           "## 6. RQ4：诊断轨迹提供什么解释线索", "",
           "### 6.1 定义与汇总口径", "",
           "PH 饱和率为全部实际来源边中 |tanh(raw)|≥0.99 的比例，每 25 轮合并。raw 是经过现有数值保护的 GP 输出。TR 熵来自固定的 4 只蚂蚁×4 个构造位置，只代表这些探针。来源年龄按实际强化路径的生成轮次计算。", "",
           "窗口 tanh 标准差合并了边与轮次，包含时间变化，不是纯粹的源内空间差异。实际源内重加权另由下面的 AD 衡量。", "",
           r"\[AD=\frac{\sum_e|D_{GP}(e)-D_0(e)|}{B}.\]", "",
           "AD 比较相同实际来源、相同总预算 B 下的 GP 沉积与均匀来源内沉积。本八格 MMAS 每轮只有一个实际来源。AD 是保存来源上的局部 PH 变化，不是另一条 baseline 轨迹的因果效应。LS 保留率为同一只蚂蚁的构造边在 2-opt 后仍保留的比例，不是 baseline 与 GP 之间的差异存活率。", "",
           markdown_table(gp_mechanisms,[("condition","条件"),("source_age","来源年龄"),("source_switch_fraction","来源切换率"),
                                        ("ph_saturation","PH 饱和率"),("ph_tanh_std","tanh 标准差"),("deposit_ad_same_source_budget","AD"),
                                        ("ls_gain_pp","2-opt 平均改善 (pp)"),("ls_retention","构造边保留率"),("restarts_per_solve","每求解重启数")]),"",
           "表中先在每次求解内平均窗口或固定采样点，再平均 seeds、实例和冠军。来源重复时长列为窗口末点；不是平均一次重复事件持续时间。baseline 与三个冠军的完整分列表见 [机制汇总](mechanism_summary.csv)。", "",
           f"原配置 C111 的 GP 平均 PH 饱和率为 {native_gp['ph_saturation']:.5f}，同来源预算下 AD 为 {native_gp['deposit_ad_same_source_budget']:.5f}。"
           f"移除 H 后，C110 的对应值为 {no_history_gp['ph_saturation']:.5f} 和 {no_history_gp['deposit_ad_same_source_budget']:.5f}。"
           "两者分别描述输出饱和和实际沉积变化，不能互相替代，也不能把跨轨迹变化解释成单独修正某个 terminal 的效果。", "",
           f"C111 的 GP 路径经 2-opt 后，平均保留 {100*native_gp['ls_retention']:.2f}% 的构造边，平均路径 gap 降低 {native_gp['ls_gain_pp']:.5f} pp。"
           "这是单条轨迹内部的局部搜索改变量。它不测量 GP 与 baseline 的结构差异被消除了多少，因此不能据此声称 2-opt 抹除了学习信号。", "",
           markdown_table(h_process,[("metric","过程量"),("native_base","baseline C111"),("no_h_base","baseline C110"),
                                     ("native_gp","GP C111"),("no_h_gp","GP C110")]),"",
           "移除 H 后，baseline 的构造质量、探针熵和 LS 工作后的路径质量变化更明显；当前 GP 的这些均值较稳定。"
           "这与 baseline 绝对退化更多的最终结果一致。但这些是受干预后共同变化的过程量，尚不能确定哪个量是最终质量差的中介原因。", "",
           "### 6.2 全八格过程图", "",
           "![PH 饱和轨迹](mechanism_ph_saturation.png)","",
           "![归一化沉积变化](mechanism_deposit_ad_same_source_budget.png)","",
           "![来源年龄](mechanism_source_age.png)","",
           "![机制总览](mechanism_overview.png)","",
           "![各冠军 PH](champion_ph_saturation.png)","",
           "图 9–13：PH 输出、实际沉积、来源及 LS 过程。相同饱和率不代表相同沉积；源内归一化可以消除共同倍率。来源年龄、来源重复与熵的变化可以定位后续干预，但不能单独证明同一搜索盆地或有害停滞。MMAS-81001 和 81003 的 TR 为零树，因此不能把它们的全部收益差归因于学到的构造规则。", "",
           "### 6.3 重启与缺失窗口", "",
           markdown_table([r for r in event_summary if r["model"]=="baseline"],[("condition","条件"),("events","baseline 重启事件"),
                           ("instances_with_events","涉及实例"),("valid_events_100","100轮有效事件"),
                           ("valid_instances_100","100轮有效实例"),("conditional_change_100_pp","100轮后变化 (pp)")]),"",
           "事件后变化先按求解、再按实例聚合，且只条件化于存在有效事件的实例。超过 5000 轮边界的窗口标为缺失，不补零。事件前后的改善不是重启的因果效应，因为没有固定时刻的无重启分叉。完整记录见 [重启事件](restart_events.csv) 与 [分模型汇总](restart_summary.csv)。", "",
           "本轮未运行完整同状态分叉、TR-only/PH-only 全因子、重启回放或 AS 反向加回机制。因此不报告这些尚不存在的机制实验结果。", "",
           "## 7. RQ5：审计成本与运行效率", "",
           "### 7.1 同卡预热后的 100 轮验收", "",
           markdown_table(validation,[("gpu_model","型号"),("variant","算法"),("mode","统计模式"),("off_seconds","无审计 (s)"),
                                      ("audit_seconds","完整审计 (s)"),("audit_wall_ratio","耗时倍数"),("oracle","输入 oracle")]),"",
           "这些是 32 实例×baseline/三个冠军的短跑，两个记录配置已预热。完整审计具有明显成本，不能声称开销小于 5%。单次短跑不是稳定的性能排名，也不能据此声称统计 FP64 普遍快于 FP32。", "",
           "### 7.2 5000 轮任务计时", "",
           markdown_table(tables["timing"],[("cohort","组"),("gpu_model","型号"),("variant","算法"),("mode","统计模式"),
                                          ("instances","实例/任务"),("tasks","任务数"),("complete_timed_tasks","完整计时数"),
                                          ("resumed_or_incomplete_timing","恢复/不完整计时"),("median_seconds","完整计时中位数 (s)")]),"",
           "最终 wall_seconds 可能只覆盖恢复后的最后一次执行。只有所有执行分块的阶段计时覆盖第 1–5000 轮时，才进入完整计时中位数。其他记录保留但不混入完整耗时。阶段计时和后台压缩可能重叠，不能简单相加。也不把批量 wall time 除以冠军数当作单模型部署延迟。", "",
           "P1 计时跨八格条件汇总，仅用于描述本批工作负载；不是单一算法配置的 benchmark。不同型号分组，不混算硬件加速比。完整审计和首次编译等开销不冒充无审计部署性能。迁卡前记录及资源暂停另存归档，不当成新的独立重复。", "",
           "## 8. 结论、证据缺口与后续用途", "",
           markdown_table([
               {"q":"H 是否影响当前 GP 的边际收益？","a":"开发集历史实现的配对干预支持；相对变化主要来自 baseline 退化更多。"},
               {"q":"R/F 可以认为无用吗？","a":"不能。单项区间不足以排除作用，且存在背景依赖与交互。"},
               {"q":"数值缺陷是否已证明是 MMAS 收益小的主因？","a":"没有。修正通过输入核验，但旧冠军质量未改善；仍需稳定环境重训。"},
               {"q":"历史强化是否与 GP 功能重叠？","a":"尚不能证明。需要 TR/PH 分解、同状态干预或反向加回机制。"},
               {"q":"GP 是否可替代 2-opt 或复制 MMAS 框架优势？","a":"本轮没有对应的无 LS 学习和框架移植实验，不能回答。"},
               {"q":"能否证明 residual 优于直接重新设计状态转移？","a":"本轮没有直接替换规则的训练对照，不能据此比较两种学习形式。"},
               {"q":"能否泛化到其他数据与重新训练？","a":"不能。独立确认集、OOD、horizon 与重新训练尚缺相应证据。"},
           ],[("q","问题"),("a","当前可支持的回答")]),"",
           "本报告可以用于确定下一轮受控实验的优先级：先在数值稳定环境中重新训练并重新验收固定冠军，再验证历史来源的背景依赖和 TR/PH 贡献。若要解释 AS 与 MMAS 的差异，需要反向机制验证。不能仅因本轮观察到较大效应，就跳过独立确认或按结果选择数据。以上为后续研究建议，本次没有启动这些实验。", "",
           "## 附录：复现与文件索引", "",
           f"冻结求解源码：`{meta['source']}`；快照登记提交：`{meta['frozen_commit']}`。分析版本及文件 SHA256 见 [provenance.json](provenance.json)。源码快照的内容哈希优先于运行时工作树 HEAD。", "",
           f"本报告核验了 {provenance['diagnostic_integrity']['jobs']} 个质量子任务的诊断索引，共 {provenance['diagnostic_integrity']['files']} 个引用文件，"
           f"{provenance['diagnostic_integrity']['bytes']/1e9:.2f} GB。该核验检查覆盖和文件完整性，不表示重新执行了所有数学 oracle。", "",
           f"另核验 18 个数值验收任务的正式诊断，共 {provenance['validation_integrity']['files']} 个文件，"
           f"{provenance['validation_integrity']['bytes']/1e9:.2f} GB；见 [验收日志完整性](validation_integrity.json)。预热记录不进入质量样本数。", "",
           "- [统计参数与完整结果](statistics.json)、[逐求解质量](quality_per_solve.csv)、[八格统计](factorial.csv)。",
           "- [数值验收](numeric_validation.csv)、[逐任务计时](timing_per_task.csv)、[完整诊断校验](integrity_audit.json)。",
           "- [anytime 均值与区间](anytime_curves.csv)、[25轮窗口轨迹](mechanism_windows.csv)、[输出文件校验](report_manifest.json)。", "",
           "```bash", "CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=src \\",
           "  .venv/bin/python -m control_experiments.mmas_ls.research_report --workers 4", "```", "",
           "首次执行读取完整诊断日志。后续仅当原始文件大小、mtime、ctime、索引和分析代码身份未变时复用校验缓存。派生机制缓存绑定诊断索引及分析代码。原始记录、旧门禁和冻结任务不会被重写。", ""]
    (target/"report_zh.md").write_text("\n".join(lines))
