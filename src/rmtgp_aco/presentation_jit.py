"""生成式 individual JIT 微基准及具有真实唯一程序数的规模扫描。"""

from __future__ import annotations

import hashlib
import random
from copy import deepcopy
from pathlib import Path
from time import perf_counter

import numpy as np
from deap import gp
from numba import njit, types
from numba.extending import register_jitable

from .aco_numba import _encode_program, _evaluate_program
from .genetic import compile_individual, valid_size
from .presentation_bench import (
    TraceWriter,
    atomic_json,
    experiment,
    load_case,
    load_trace,
    pack_individual,
    primitive_sets,
    read_json,
    write_csv,
)
from .program import ERCValue, compile_tree


@register_jitable
def scalar_sanitize(value):
    """与 CPU postfix 的逐节点保护一致，Python 与 Numba 共用。"""
    if np.isnan(value):
        return 0.0
    if value > 10.0:
        return 10.0
    if value < -10.0:
        return -10.0
    return value


def scalar_function(program, role):
    """从同一 postfix 程序生成直线数值代码，不调用结构 marker。"""
    encoded = _encode_program(program, role=role)
    lines = ["def individual(x, column):"]
    stack = []
    for index, instruction in enumerate(program.instructions):
        name = f"v{index}"
        opcode = instruction.opcode
        if opcode == "CONST":
            expression = repr(float(instruction.argument))
        elif opcode == "TERMINAL":
            expression = f"x[{int(encoded.integer_arguments[index])}, column]"
        elif opcode in ("ABS", "NEG"):
            left = stack.pop()
            expression = f"abs({left})" if opcode == "ABS" else f"-{left}"
        else:
            right, left = stack.pop(), stack.pop()
            expressions = {
                "ADD": f"{left} + {right}",
                "SUB": f"{left} - {right}",
                "MUL": f"{left} * {right}",
                "PDIV": f"{left} * {right} / ({right} * {right} + 1e-6)",
                "PDIV1": f"({left} / {right} if abs({right}) > 1e-6 else 1.0)",
                "MIN": f"min({left}, {right})",
                "MAX": f"max({left}, {right})",
            }
            expression = expressions[opcode]
        if opcode not in ("CONST", "TERMINAL"):
            expression = f"scalar_sanitize({expression})"
        lines.append(f"    {name} = {expression}")
        stack.append(name)
    lines.append(f"    return scalar_sanitize({stack[0]})")
    source = "\n".join(lines) + "\n"
    namespace = {"scalar_sanitize": scalar_sanitize}
    exec(compile(source, "<presentation-individual>", "exec"), namespace)
    return namespace["individual"], source, encoded


def sum_loop(function):
    """每次输入都随 column 改变，返回累加值，防止无用计算被删除。"""

    def loop(inputs, count):
        total = 0.0
        for index in range(count):
            total += function(inputs, index % inputs.shape[1])
        return total

    return loop


def varied_inputs(encoded):
    """固定且随列变化的输入；选择树与正式微基准共用同一输入分布。"""
    rng = np.random.default_rng(20260915)
    terminals = max(1, int(encoded.integer_arguments.max()) + 1)
    inputs = np.ascontiguousarray(rng.uniform(-2, 2, (terminals, 4096)))
    inputs[:, ::31] = 0
    return inputs


def has_variable_output(program, role):
    """排除常数树及输入相消树，防止编译器折叠掉整个表达式。"""
    pure, _, encoded = scalar_function(program, role)
    inputs = varied_inputs(encoded)
    values = np.array([pure(inputs, index) for index in range(128)])
    return bool(np.isfinite(values).all() and np.ptp(values) > 1e-12)


@njit(cache=True)
def postfix_sum(opcodes, floats, integers, inputs, count, stack_size):
    stack = np.empty(stack_size, dtype=np.float64)
    total = 0.0
    for index in range(count):
        total += _evaluate_program(
            opcodes, floats, integers, inputs, index % inputs.shape[1], stack
        )
    return total


def run_jit(output: Path, destination: Path):
    exp = experiment(output, "cpu1")
    calls = load_trace(output / "trace", exp.gp)
    trees = {}
    by_generation = {}
    for call, population, _ in calls:
        for individual in population:
            for role, tree in zip(("transition", "pheromone"), individual, strict=True):
                program = compile_tree(tree, role=role)
                key = hashlib.sha256((role + str(tree)).encode()).hexdigest()
                if key not in trees:
                    trees[key] = (tree, role, program)
                    by_generation[key] = call["generation"]
    eligible = sorted(
        (
            k
            for k in trees
            if len(trees[k][0]) > 1 and has_variable_output(trees[k][2], trees[k][1])
        ),
        key=lambda k: (len(trees[k][0]), k),
    )
    if len(eligible) < 8:
        raise ValueError("真实 trace 中不足 8 棵具有变化输出的非平凡树")
    selection = [eligible[int(i)] for i in np.linspace(0, len(eligible) - 1, 8)]
    rows, compilation_rows = [], []
    sources = {}
    loop_signature = (types.Array(types.float64, 2, "C"), types.int64)
    for repeat in range(3):
        for key in selection:
            tree, role, program = trees[key]
            code_started = perf_counter()
            pure, source, encoded = scalar_function(program, role)
            codegen = perf_counter() - code_started
            sources[key] = source
            inputs = varied_inputs(encoded)
            python_loop = sum_loop(pure)
            compiled = njit(fastmath=False)(sum_loop(njit(inline="always", fastmath=False)(pure)))
            compile_started = perf_counter()
            compiled.compile(loop_signature)
            compile_s = perf_counter() - compile_started
            arguments = (
                encoded.opcodes,
                encoded.float_arguments,
                encoded.integer_arguments,
                inputs,
            )
            postfix_sum(*arguments, 512, encoded.stack_size)
            expected = python_loop(inputs, 512)
            if not np.isclose(expected, compiled(inputs, 512), rtol=1e-12, atol=1e-12):
                raise AssertionError("individual JIT 与 Python 数值不一致")
            if not np.isclose(
                expected, postfix_sum(*arguments, 512, encoded.stack_size), rtol=1e-12, atol=1e-12
            ):
                raise AssertionError("生成函数与 postfix 数值不一致")
            for count in (100, 1000, 10000, 100000, 1000000):
                for method in ("python", "individual_jit", "postfix_numba"):
                    started = perf_counter()
                    if method == "python":
                        checksum = python_loop(inputs, count)
                    elif method == "individual_jit":
                        checksum = compiled(inputs, count)
                    else:
                        checksum = postfix_sum(*arguments, count, encoded.stack_size)
                    execute_s = perf_counter() - started
                    rows.append(
                        {
                            "program_hash": key,
                            "role": role,
                            "node_count": len(tree),
                            "input_hash": hashlib.sha256(inputs.tobytes()).hexdigest(),
                            "dtype": "fp64",
                            "method": method,
                            "call_count": count,
                            "repeat_id": repeat,
                            "cache_state": "compiled_execution",
                            "codegen_s": codegen,
                            "compile_s": compile_s if method == "individual_jit" else None,
                            "execute_s": execute_s,
                            "total_s": execute_s
                            + (compile_s + codegen if method == "individual_jit" else 0),
                            "checksum": float(checksum),
                            "compiled_signatures": len(compiled.signatures),
                            "status": "completed",
                        }
                    )
            write_csv(destination / "jit_microbench.csv", rows)
            print(f"jit repeat={repeat} nodes={len(tree)} compile={compile_s:.3f}s", flush=True)
    # 所有新增树单独计时，不把它们的缓存预热混入上面的微基准。
    for key, (tree, role, program) in trees.items():
        started = perf_counter()
        pure, _, _ = scalar_function(program, role)
        compiled = njit(fastmath=False)(sum_loop(njit(inline="always")(pure)))
        compiled.compile(loop_signature)
        compilation_rows.append(
            {
                "program_hash": key,
                "generation": by_generation[key],
                "node_count": len(tree),
                "compile_codegen_s": perf_counter() - started,
            }
        )
        if len(compilation_rows) % 20 == 0:
            print(f"compiled_new_trees={len(compilation_rows)}/{len(trees)}", flush=True)
            write_csv(destination / "new_tree_compiles.csv", compilation_rows)
    write_csv(destination / "new_tree_compiles.csv", compilation_rows)
    atomic_json(destination / "sources.json", sources)
    atomic_json(
        destination / "result.json", {"kind": "jit", "records": rows, "new_trees": compilation_rows}
    )


def prepare_scans(output: Path):
    """固定嵌套程序集；每一点均验证后端语义去重后的实际基数。"""
    from .aco_cuda import _active_and_representative_programs

    exp = experiment(output, "v2")
    calls = load_trace(output / "trace", exp.gp)
    source_call, initial, cases = next(c for c in calls if c[0]["generation"] == 3)
    psets = primitive_sets(exp.gp)
    random.seed(2026091504)
    population = []
    programs = []
    structural = set()
    candidates = [p for p in initial if p.total_nodes > 1]
    attempts = 0
    while len(population) < 256:
        attempts += 1
        if attempts > 100000:
            raise RuntimeError("无法在预算内生成 256 个 backend-unique programs")
        candidate = random.choice(candidates).clone()
        if attempts > len(candidates):
            role = random.randrange(2)
            tree = candidate[role]
            index = random.randrange(len(tree))
            node = tree[index]
            if isinstance(node, gp.Primitive):
                choices = [
                    p for p in psets[role].primitives[psets[role].ret] if p.arity == node.arity
                ]
                tree[index] = deepcopy(random.choice(choices))
            elif isinstance(node.value, ERCValue):
                tree[index] = gp.Terminal(ERCValue(random.uniform(-1, 1)), False, psets[role].ret)
            else:
                choices = [
                    t for t in psets[role].terminals[psets[role].ret] if not isinstance(t, type)
                ]
                tree[index] = deepcopy(random.choice(choices))
        if not valid_size(candidate, exp.gp) or candidate.structural_hash in structural:
            continue
        compiled = compile_individual(candidate)
        representatives = _active_and_representative_programs(programs + [compiled], exp.aco)[4]
        if len(representatives) != len(programs) + 1:
            continue
        structural.add(candidate.structural_hash)
        population.append(candidate)
        programs.append(compiled)
    manifest = read_json(output / "workload_manifest.json")
    scan_rows = []
    for count, scale in [(p, 100) for p in (1, 8, 32, 100, 256)] + [(100, 50), (100, 500)]:
        case = (
            cases[0]
            if scale == 100
            else load_case(
                output / "inputs", next(d for d in manifest["cases"] if d["scale"] == scale)
            )
        )
        name = f"p{count}-n{scale}"
        writer = TraceWriter(output / "scans" / name, exp.aco)
        writer.capture(3, population[:count], [case])
        scan_rows.append(
            {
                "name": name,
                "population": count,
                "cities": scale,
                "source_workload": source_call["workload_id"],
                "workload_id": writer.calls[0]["workload_id"],
                "nodes_mean": float(np.mean([p.total_nodes for p in population[:count]])),
            }
        )
    atomic_json(
        output / "scans/manifest.json",
        {
            "scans": scan_rows,
            "generation3_nodes_mean": float(np.mean([p.total_nodes for p in candidates])),
            "program_pool": [pack_individual(p) for p in population],
        },
    )
