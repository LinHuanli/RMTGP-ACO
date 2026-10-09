"""输出可审阅补丁：从锁定旧源文件移植 CPU 构造器，显式修正 FP32。

该脚本不自行写源码。其输出通过 apply_patch 应用，保留来源和改动记录。
"""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = (ROOT / "references/gpaco_2b847698/aco_numba.py").read_text()
names = {
    "_mask_has",
    "_sanitize",
    "_softplus_clipped",
    "_evaluate_program",
    "_evaluate_program_columns",
    "_nearest_neighbour_length",
    "_prepare_transition_scores",
    "_choose_city",
    "_apply_acs_edges",
    "_construct_tours",
    "_tour_lengths",
}


class Float32(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, float):
            return ast.Call(
                ast.Attribute(ast.Name("np", ast.Load()), "float32", ast.Load()), [node], []
            )
        return node

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == "float":
            node.func = ast.Attribute(ast.Name("np", ast.Load()), "float32", ast.Load())
        return node


tree = ast.parse(source)
selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
body = ast.unparse(
    ast.fix_missing_locations(Float32().visit(ast.Module(body=selected, type_ignores=[])))
)
# 整数和 float32 混算会被 Numba 隐式提升；逐处固定数值域。
for old, new in [
    ("np.float32(1.0) / count", "np.float32(1.0) / np.float32(count)"),
    ("np.float32(2.0) * index", "np.float32(2.0) * np.float32(index)"),
    ("np.float32(2.0) * rank", "np.float32(2.0) * np.float32(rank)"),
    ("np.float32(2.0) * construction_step", "np.float32(2.0) * np.float32(construction_step)"),
    ("/ max(distances.shape[0] - 1, 1)", "/ np.float32(max(distances.shape[0] - 1, 1))"),
    ("np.float32(2.0) * (iteration - 1)", "np.float32(2.0) * np.float32(iteration - 1)"),
    ("/ max(total_iterations - 1, 1)", "/ np.float32(max(total_iterations - 1, 1))"),
    ("stagnation / total_iterations", "np.float32(stagnation) / np.float32(total_iterations)"),
    ("mean_tau /= count", "mean_tau /= np.float32(count)"),
    ("mean_distance /= count", "mean_distance /= np.float32(count)"),
    ("roulette_uniform * count", "roulette_uniform * np.float32(count)"),
    ("start_uniform * n", "start_uniform * np.float32(n)"),
    ("score = tau * (eta * eta)", "score = (tau * eta) * eta"),
    ("best_distance = np.inf", "best_distance = np.float32(np.inf)"),
]:
    body = body.replace(old, new)
header = '''"""从 2b847698 移植的单树 CPU 构造器；不含信息素树或局部搜索。

保留实例内部紧凑候选、标量广播和向量列解释器作为强基线。
浮点常量和整数比例显式 FP32；随机流和稳定方差由共享规范替换。
"""
import numpy as np
from numba import njit
from .numeric import counter_uniform as _counter_uniform, stdrel as _masked_stdrel_values

_CONST, _TERMINAL, _ADD, _SUB, _MUL, _PDIV, _PDIV1, _MIN, _MAX, _ABS, _NEG = range(11)
_TRANSITION_SCALAR_TERMINAL_MASK = np.uint64(sum(1 << x for x in (4,5,6,7,10,11,12,13)))
'''
output = header + "\n" + body + "\n"
print("*** Begin Patch\n*** Add File: " + str(ROOT / "src/gpaco/backends/cpu_existing.py"))
print("\n".join("+" + line for line in output.splitlines()))
print("*** End Patch")
