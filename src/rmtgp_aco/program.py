"""Strongly Typed GP primitive sets 与 PyTorch postfix interpreter。"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import partial

import torch
from deap import gp


class TrField:
    """DEAP 类型标记：transition candidate field。"""


class PhField:
    """DEAP 类型标记：pheromone event-edge field。"""


@dataclass(frozen=True, slots=True)
class SymbolRef:
    """运行时从 context 字典读取的符号 terminal。"""

    name: str

    def __repr__(self) -> str:
        return self.name


@dataclass(slots=True)
class ERCValue:
    """可被 point mutation 扰动的 ephemeral random constant。"""

    value: float

    def __repr__(self) -> str:
        return f"{self.value:.17g}"


def _make_erc() -> ERCValue:
    """DEAP ephemeral constant 工厂，使用 GP 层控制的 Python RNG。"""

    return ERCValue(random.uniform(-1.0, 1.0))


def _marker_add(left: object, right: object) -> object:
    return left


def _marker_sub(left: object, right: object) -> object:
    return left


def _marker_mul(left: object, right: object) -> object:
    return left


def _marker_pdiv(left: object, right: object) -> object:
    return left


def _marker_min(left: object, right: object) -> object:
    return left


def _marker_max(left: object, right: object) -> object:
    return left


def _marker_abs(value: object) -> object:
    return value


def _marker_neg(value: object) -> object:
    return value


TRANSITION_TERMINALS: tuple[str, ...] = (
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

CORE_TRANSITION_TERMINALS: tuple[str, ...] = (
    "RTau",
    "REta",
    "BaseConf",
    "DistRank",
)

PHEROMONE_TERMINALS: tuple[str, ...] = (
    "EdgeEta",
    "EdgeTau",
    "NNRank",
    "ColonyFreq",
    "SourceQuality",
    "ACOProg",
    "Stagnation",
)

# ``LSGain`` 的数值语义由 ACOConfig.ls_gain_semantics 显式控制。旧实验
# 默认读取 tour-level 常数；LS-v2 配置使用逐边最后一次引入增益。
LEGACY_LS_PHEROMONE_TERMINAL = "LSGain"
ORIGIN_PHEROMONE_TERMINALS: tuple[str, ...] = PHEROMONE_TERMINALS + ("Origin",)
ALLOWED_PHEROMONE_TERMINALS: tuple[str, ...] = (
    *ORIGIN_PHEROMONE_TERMINALS,
    LEGACY_LS_PHEROMONE_TERMINAL,
    "TauHeadroom",
    "PreFreq",
    "PostFreq",
)

CORE_PHEROMONE_TERMINALS: tuple[str, ...] = (
    "EdgeEta",
    "EdgeTau",
    "NNRank",
    "SourceQuality",
)

LEGACY_TRANSITION_TERMINALS: tuple[str, ...] = (
    "Tau",
    "Distance",
    "MeanTau",
    "MeanDistance",
    "Size",
    "FeasibleCount",
)

NAMED_CONSTANTS: dict[str, float] = {
    "NEG_ONE": -1.0,
    "NEG_HALF": -0.5,
    "ZERO_TR": 0.0,
    "ZERO_PH": 0.0,
    "POS_HALF": 0.5,
    "POS_ONE": 1.0,
}


def _add_typed_primitives(
    pset: gp.PrimitiveSetTyped,
    field_type: type,
    *,
    function_profile: str,
) -> None:
    """向指定 field type 注册同构的逐元素函数集。"""

    pset.addPrimitive(_marker_add, [field_type, field_type], field_type, name="ADD")
    pset.addPrimitive(_marker_sub, [field_type, field_type], field_type, name="SUB")
    pset.addPrimitive(_marker_mul, [field_type, field_type], field_type, name="MUL")
    pset.addPrimitive(_marker_pdiv, [field_type, field_type], field_type, name="PDIV")
    pset.addPrimitive(_marker_neg, [field_type], field_type, name="NEG")
    if function_profile == "f1":
        pset.addPrimitive(_marker_min, [field_type, field_type], field_type, name="MIN")
        pset.addPrimitive(_marker_max, [field_type, field_type], field_type, name="MAX")
        pset.addPrimitive(_marker_abs, [field_type], field_type, name="ABS")
    elif function_profile != "f0":
        raise ValueError(f"未知 function profile: {function_profile}")


def _add_legacy_primitives(
    pset: gp.PrimitiveSetTyped,
    field_type: type,
) -> None:
    """上一研究的简化 function set；除零时返回 1。"""

    pset.addPrimitive(_marker_add, [field_type, field_type], field_type, name="ADD")
    pset.addPrimitive(_marker_sub, [field_type, field_type], field_type, name="SUB")
    pset.addPrimitive(_marker_mul, [field_type, field_type], field_type, name="MUL")
    pset.addPrimitive(
        _marker_pdiv,
        [field_type, field_type],
        field_type,
        name="PDIV1",
    )
    pset.addPrimitive(_marker_neg, [field_type], field_type, name="NEG")


def _add_typed_terminals(
    pset: gp.PrimitiveSetTyped,
    field_type: type,
    names: Sequence[str],
    zero_name: str,
) -> None:
    """注册 context terminals、精确常数和 ERC。"""

    for name in names:
        pset.addTerminal(SymbolRef(name), field_type, name=name)
    for label, value in (
        ("NEG_ONE", -1.0),
        ("NEG_HALF", -0.5),
        (zero_name, 0.0),
        ("POS_HALF", 0.5),
        ("POS_ONE", 1.0),
    ):
        pset.addTerminal(ERCValue(value), field_type, name=label)
    # partial 是可 pickle 的，避免 DEAP 对 lambda ephemeral constant 的警告。
    pset.addEphemeralConstant(
        f"ERC_{zero_name}",
        partial(_make_erc),
        field_type,
    )


def create_primitive_sets(
    *,
    transition_profile: str = "main",
    function_profile: str = "f1",
    transition_terminals: Sequence[str] | None = None,
    pheromone_terminals: Sequence[str] | None = None,
) -> tuple[gp.PrimitiveSetTyped, gp.PrimitiveSetTyped]:
    """建立相互类型隔离的 transition 与 pheromone primitive sets。"""

    transition = gp.PrimitiveSetTyped("TRANSITION", [], TrField)
    pheromone = gp.PrimitiveSetTyped("PHEROMONE", [], PhField)
    if transition_profile == "main":
        _add_typed_primitives(
            transition,
            TrField,
            function_profile=function_profile,
        )
        selected_transition_terminals = (
            TRANSITION_TERMINALS
            if transition_terminals is None
            else tuple(transition_terminals)
        )
        invalid = set(selected_transition_terminals) - set(TRANSITION_TERMINALS)
        if invalid:
            raise ValueError(f"未知 transition terminals: {sorted(invalid)}")
    elif transition_profile == "legacy":
        _add_legacy_primitives(transition, TrField)
        selected_transition_terminals = (
            LEGACY_TRANSITION_TERMINALS
            if transition_terminals is None
            else tuple(transition_terminals)
        )
        invalid = set(selected_transition_terminals) - set(
            LEGACY_TRANSITION_TERMINALS
        )
        if invalid:
            raise ValueError(f"未知 legacy terminals: {sorted(invalid)}")
    else:
        raise ValueError(f"未知 transition profile: {transition_profile}")
    selected_pheromone_terminals = (
        PHEROMONE_TERMINALS
        if pheromone_terminals is None
        else tuple(pheromone_terminals)
    )
    invalid_pheromone = set(selected_pheromone_terminals) - set(
        ALLOWED_PHEROMONE_TERMINALS
    )
    if invalid_pheromone:
        raise ValueError(f"未知 pheromone terminals: {sorted(invalid_pheromone)}")
    _add_typed_primitives(
        pheromone,
        PhField,
        function_profile=function_profile,
    )
    _add_typed_terminals(
        transition,
        TrField,
        selected_transition_terminals,
        "ZERO_TR",
    )
    _add_typed_terminals(
        pheromone,
        PhField,
        selected_pheromone_terminals,
        "ZERO_PH",
    )
    return transition, pheromone


@dataclass(frozen=True, slots=True)
class Instruction:
    """一个 postfix 指令。"""

    opcode: str
    argument: str | float | None = None


@dataclass(frozen=True, slots=True)
class TensorProgram:
    """可在 PyTorch tensor 上执行的 GP postfix program。"""

    instructions: tuple[Instruction, ...]
    role: str
    expression: str
    required_terminals: frozenset[str]

    @property
    def is_exact_zero(self) -> bool:
        """该程序是否为用于恢复原始 ACO 的单节点常数零树。"""

        return (
            len(self.instructions) == 1
            and self.instructions[0].opcode == "CONST"
            and float(self.instructions[0].argument) == 0.0
        )

    def evaluate(
        self,
        context: Mapping[str, torch.Tensor],
        *,
        template: torch.Tensor | None = None,
        epsilon_division: float = 1e-6,
        clip_value: float = 10.0,
    ) -> torch.Tensor:
        """执行程序，并在每个 primitive 后进行 finite/clip 保护。"""

        if template is None:
            if not context:
                raise ValueError("常数 program 需要显式 template")
            template = next(iter(context.values()))
        stack: list[torch.Tensor] = []

        def sanitize(value: torch.Tensor) -> torch.Tensor:
            value = torch.nan_to_num(
                value,
                nan=0.0,
                posinf=clip_value,
                neginf=-clip_value,
            )
            return torch.clamp(value, -clip_value, clip_value)

        for instruction in self.instructions:
            opcode = instruction.opcode
            if opcode == "TERMINAL":
                name = str(instruction.argument)
                try:
                    stack.append(context[name])
                except KeyError as exc:
                    raise KeyError(f"{self.role} context 缺少 terminal: {name}") from exc
                continue
            if opcode == "CONST":
                stack.append(torch.full_like(template, float(instruction.argument)))
                continue

            if opcode in {"ABS", "NEG"}:
                if not stack:
                    raise RuntimeError("非法 postfix program：一元运算缺少 operand")
                value = stack.pop()
                result = torch.abs(value) if opcode == "ABS" else -value
                stack.append(sanitize(result))
                continue

            if len(stack) < 2:
                raise RuntimeError("非法 postfix program：二元运算缺少 operands")
            right = stack.pop()
            left = stack.pop()
            if opcode == "ADD":
                result = left + right
            elif opcode == "SUB":
                result = left - right
            elif opcode == "MUL":
                result = left * right
            elif opcode == "PDIV":
                result = left * right / (right * right + epsilon_division)
            elif opcode == "PDIV1":
                result = torch.where(
                    torch.abs(right) > epsilon_division,
                    left / right,
                    torch.ones_like(left),
                )
            elif opcode == "MIN":
                result = torch.minimum(left, right)
            elif opcode == "MAX":
                result = torch.maximum(left, right)
            else:
                raise RuntimeError(f"未知 GP opcode: {opcode}")
            stack.append(sanitize(result))

        if len(stack) != 1:
            raise RuntimeError(f"非法 postfix program：结束时 stack size={len(stack)}")
        return sanitize(stack[0])


def _terminal_instruction(node: gp.Terminal) -> Instruction:
    """把 DEAP terminal 转换为 context lookup 或常数。"""

    value = node.value
    if isinstance(value, SymbolRef):
        return Instruction("TERMINAL", value.name)
    if isinstance(value, ERCValue):
        return Instruction("CONST", float(value.value))
    if isinstance(value, (int, float)):
        return Instruction("CONST", float(value))

    # 命名 terminal 在 DEAP 中可能以 context key 的字符串形式保存。
    if isinstance(value, str):
        if (
            value in TRANSITION_TERMINALS
            or value in LEGACY_TRANSITION_TERMINALS
            or value in ALLOWED_PHEROMONE_TERMINALS
        ):
            return Instruction("TERMINAL", value)
        if value in NAMED_CONSTANTS:
            return Instruction("CONST", NAMED_CONSTANTS[value])
    if node.name in NAMED_CONSTANTS:
        return Instruction("CONST", NAMED_CONSTANTS[node.name])
    raise TypeError(f"不支持的 GP terminal: name={node.name!r}, value={value!r}")


def compile_tree(tree: gp.PrimitiveTree, *, role: str) -> TensorProgram:
    """把 DEAP 的 prefix tree 确定性编译为 postfix program。"""

    instructions: list[Instruction] = []

    def visit(position: int) -> int:
        node = tree[position]
        if node.arity == 0:
            instructions.append(_terminal_instruction(node))
            return position + 1
        next_position = position + 1
        for _ in range(node.arity):
            next_position = visit(next_position)
        instructions.append(Instruction(node.name))
        return next_position

    end = visit(0)
    if end != len(tree):
        raise ValueError("GP tree 含未被访问的节点")
    return TensorProgram(
        instructions=tuple(instructions),
        role=role,
        expression=str(tree),
        required_terminals=frozenset(
            str(instruction.argument)
            for instruction in instructions
            if instruction.opcode == "TERMINAL"
        ),
    )


def constant_zero_tree(pset: gp.PrimitiveSetTyped, terminal_name: str) -> gp.PrimitiveTree:
    """构造可精确恢复 baseline 的常数零树。"""

    return gp.PrimitiveTree([pset.mapping[terminal_name]])


def perturb_erc(
    tree: gp.PrimitiveTree,
    *,
    sigma: float = 0.1,
) -> tuple[gp.PrimitiveTree]:
    """随机扰动一个 ERC；若树中没有 ERC，则保持不变。"""

    candidates = [
        node
        for node in tree
        if isinstance(node, gp.Terminal)
        and isinstance(node.value, ERCValue)
        and node.name.startswith("ERC_")
    ]
    if not candidates:
        return (tree,)
    node = random.choice(candidates)
    node.value.value = max(-1.0, min(1.0, node.value.value + random.gauss(0.0, sigma)))
    return (tree,)


def expression_is_finite(tree: gp.PrimitiveTree) -> bool:
    """检查 ERC 是否全部有限，便于遗传算子后快速拒绝坏个体。"""

    return all(
        not isinstance(node, gp.Terminal)
        or not isinstance(node.value, ERCValue)
        or math.isfinite(node.value.value)
        for node in tree
    )
