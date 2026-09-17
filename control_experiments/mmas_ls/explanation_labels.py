"""读者可见名称；内部冻结编号保持不变，但不能成为正文的解释负担。"""

CONDITION_LABELS = {
    "C111": "完整 MMAS",
    "C110": "仅关闭历史路径强化",
    "C101": "仅关闭信息素下界保护",
    "C100": "关闭下界保护与历史路径强化",
    "C011": "仅关闭重启",
    "C010": "关闭重启与历史路径强化",
    "C001": "关闭重启与下界保护",
    "C000": "关闭上述三个组件",
}
COMPONENTS = ("重启", "信息素下界保护", "历史路径强化")
MODEL_LABELS = {"baseline": "不使用 GP", "GP_mean": "三个 GP 表达式均值",
                "81001": "第一个 GP 表达式", "81002": "第二个 GP 表达式", "81003": "第三个 GP 表达式"}
PERIODS = ((1, 250), (251, 1000), (1001, 2500), (2501, 5000))


def component_columns(condition):
    """表格直接列出三个组件是否启用，而不是显示二进制编码。"""
    if condition not in CONDITION_LABELS:
        raise ValueError("未知冻结条件")
    return dict(zip(COMPONENTS, ("启用" if v == "1" else "关闭" for v in condition[1:])))


def benefit_interval(row, mean="mean_pp", low="ci_low_pp", high="ci_high_pp"):
    """从旧的 GP 减对照口径转为改进口径；取负号时必须交换上下限。"""
    return -float(row[mean]), -float(row[high]), -float(row[low])
