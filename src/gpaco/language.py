"""DEAP 单树语言与跨后端 postfix 表示；运行常数始终为 FP32。"""

import ast
import random
import struct
from dataclasses import dataclass
from hashlib import sha256

import numpy as np
from deap import gp


class TrField:
    """候选边场：标量终端可广播，逻辑输出为 [task,ant,candidate]。"""


TERMINALS = (
    "RTau",
    "REta",
    "BaseConf",
    "DistRank",
    "Entropy",
    "ConstructProg",
    "ACOProg",
    "Stagnation",
    "MutualRank",
    "TurnCos",
)
# 与锁定 CUDA 构造器的内部 ABI 一致；8--13 是不暴露给 GP 的旧符号槽。
TERMINAL_IDS = dict(zip(TERMINALS, (0, 1, 2, 3, 4, 5, 6, 7, 14, 15), strict=True))
OPS = {
    "ADD": (2, 2),
    "SUB": (3, 2),
    "MUL": (4, 2),
    "PDIV": (5, 2),
    "MIN": (7, 2),
    "MAX": (8, 2),
    "ABS": (9, 1),
    "NEG": (10, 1),
}
CONSTANTS = {"NEG_ONE": -1.0, "NEG_HALF": -0.5, "ZERO": 0.0, "POS_HALF": 0.5, "POS_ONE": 1.0}


def _binary(a, b):
    return a


def _unary(a):
    return a


def _erc():
    return float(np.float32(random.uniform(-1, 1)))


def primitive_set():
    pset = gp.PrimitiveSetTyped("transition", [], TrField)
    for name, (_, arity) in OPS.items():
        pset.addPrimitive(_unary if arity == 1 else _binary, [TrField] * arity, TrField, name=name)
    for name in TERMINALS:
        pset.addTerminal(name, TrField, name=name)
    for name, value in CONSTANTS.items():
        pset.addTerminal(value, TrField, name=name)
    pset.addEphemeralConstant("ERC", _erc, TrField)
    return pset


@dataclass(frozen=True)
class ProgramSpec:
    expression: str
    instructions: tuple[tuple[int, float, int], ...]
    depth: int

    @classmethod
    def from_tree(cls, tree):
        code = []

        def visit(position):
            node = tree[position]
            next_position = position + 1
            for _ in range(node.arity):
                next_position = visit(next_position)
            if node.arity:
                code.append((OPS[node.name][0], 0.0, -1))
            elif node.name in TERMINAL_IDS:
                code.append((1, 0.0, TERMINAL_IDS[node.name]))
            else:
                value = CONSTANTS.get(node.name, node.value)
                code.append((0, float(np.float32(value)), -1))
            return next_position

        visit(0)
        if len(code) > 31 or tree.height > 5:
            raise ValueError("单树超过 31 节点或深度 5")
        return cls(str(tree), tuple(code), tree.height)

    @classmethod
    def parse(cls, expression):
        return cls.from_tree(parse_tree(expression))

    @property
    def semantic_hash(self):
        payload = b"single-fp32-v1" + b"".join(struct.pack("<bfi", *i) for i in self.instructions)
        return sha256(payload).hexdigest()

    @property
    def required_mask(self):
        return sum(1 << t for t in {i[2] for i in self.instructions if i[0] == 1})

    @property
    def structural_hash(self):
        # 常数只占参数槽；结构哈希不能用于复用 fitness。
        payload = b"single-structure-v1" + b"".join(
            struct.pack("<bi", op, terminal) for op, _, terminal in self.instructions
        )
        return sha256(payload).hexdigest()

    def record(self):
        return {
            "expression": self.expression,
            "semantic_hash": self.semantic_hash,
            "structural_hash": self.structural_hash,
            "node_count": len(self.instructions),
            "depth": self.depth,
            "required_mask": self.required_mask,
            "instructions": self.instructions,
        }


def pack_programs(programs):
    """不同程序绝不因结构相同而共享适应度；这里只打包字节码。"""
    width = max(len(p.instructions) for p in programs)
    ops = np.zeros((len(programs), width), np.int8)
    fargs = np.zeros(ops.shape, np.float32)
    iargs = np.zeros(ops.shape, np.int16)
    lengths = np.zeros(len(programs), np.int16)
    masks = np.zeros(len(programs), np.uint64)
    active = np.ones(len(programs), np.uint8)
    for i, program in enumerate(programs):
        lengths[i] = len(program.instructions)
        masks[i] = program.required_mask
        for j, (op, value, terminal) in enumerate(program.instructions):
            ops[i, j], fargs[i, j], iargs[i, j] = op, value, terminal
    return ops, fargs, iargs, lengths, masks, active


def parse_tree(expression):
    """仅解析研究语言；支持 FP32 ERC 字面量，不执行 Python 表达式。"""
    pset = primitive_set()
    nodes = []

    def walk(node):
        if isinstance(node, ast.Name) and node.id in (*TERMINALS, *CONSTANTS):
            nodes.append(pset.mapping[node.id])
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in OPS:
            if node.keywords or len(node.args) != OPS[node.func.id][1]:
                raise ValueError("函数输入数不符合 typed primitive 定义")
            nodes.append(pset.mapping[node.func.id])
            for argument in node.args:
                walk(argument)
        else:
            try:
                value = ast.literal_eval(node)
            except (ValueError, TypeError) as error:
                raise ValueError("不支持的程序表达式") from error
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not np.isfinite(value)
                or not -1 <= value <= 1
            ):
                raise ValueError("ERC 必须是 [-1,1] 内的有限实数")
            nodes.append(gp.Terminal(float(np.float32(value)), False, TrField))

    walk(ast.parse(expression, mode="eval").body)
    return gp.PrimitiveTree(nodes)


def evaluate_reference(program, context, shape=None):
    """独立 NumPy FP32 解释器；逐算子保护，不作代数重关联。"""
    if shape is None:
        shape = next(iter(context.values())).shape if context else ()
    stack = []
    for op, value, terminal in program.instructions:
        if op == 0:
            stack.append(np.full(shape, value, dtype=np.float32))
            continue
        if op == 1:
            name = next(k for k, v in TERMINAL_IDS.items() if v == terminal)
            stack.append(np.broadcast_to(np.asarray(context[name], np.float32), shape))
            continue
        right = stack.pop()
        with np.errstate(all="ignore"):
            if op == 9:
                result = np.abs(right)
            elif op == 10:
                result = -right
            else:
                left = stack.pop()
                if op == 2:
                    result = left + right
                elif op == 3:
                    result = left - right
                elif op == 4:
                    result = left * right
                elif op == 5:
                    result = (left * right) / (right * right + np.float32(1e-6))
                elif op == 7:
                    result = np.minimum(left, right)
                else:
                    result = np.maximum(left, right)
        stack.append(
            np.clip(np.nan_to_num(result, nan=0.0, posinf=10.0, neginf=-10.0), -10.0, 10.0).astype(
                np.float32
            )
        )
    return np.clip(np.nan_to_num(stack[0], nan=0.0, posinf=10.0, neginf=-10.0), -10.0, 10.0).astype(
        np.float32
    )
