"""DEAP Strongly Typed 双树个体与遗传算子。"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable
from copy import deepcopy
from functools import partial
from hashlib import sha256

from deap import base, gp

from .config import GPConfig
from .program import (
    PhField,
    TensorProgram,
    TrField,
    compile_tree,
    constant_zero_tree,
    create_primitive_sets,
    expression_is_finite,
    perturb_erc,
)


class FitnessMin(base.Fitness):
    """单目标最小化 fitness。"""

    weights = (-1.0,)


class RMTGPIndividual(list):
    """包含 transition/phero 两棵 typed trees 的 DEAP 个体。"""

    def __init__(
        self,
        transition_tree: gp.PrimitiveTree,
        pheromone_tree: gp.PrimitiveTree,
    ) -> None:
        super().__init__([transition_tree, pheromone_tree])
        self.fitness = FitnessMin()
        self.metadata: dict[str, object] = {}

    @property
    def transition_tree(self) -> gp.PrimitiveTree:
        return self[0]

    @transition_tree.setter
    def transition_tree(self, tree: gp.PrimitiveTree) -> None:
        self[0] = tree

    @property
    def pheromone_tree(self) -> gp.PrimitiveTree:
        return self[1]

    @pheromone_tree.setter
    def pheromone_tree(self, tree: gp.PrimitiveTree) -> None:
        self[1] = tree

    @staticmethod
    def _effective_nodes(tree: gp.PrimitiveTree) -> int:
        """零残差哨兵不占表达能力预算，其余节点按语法树原样计数。"""

        if len(tree) == 1 and str(tree) in {"ZERO_TR", "ZERO_PH"}:
            return 0
        return len(tree)

    @property
    def transition_nodes(self) -> int:
        return self._effective_nodes(self.transition_tree)

    @property
    def pheromone_nodes(self) -> int:
        return self._effective_nodes(self.pheromone_tree)

    @property
    def structural_hash(self) -> str:
        """对两棵树表达式计算可用于 memoization 的哈希。"""

        passthrough = bool(self.metadata.get("baseline_passthrough", False))
        payload = f"{int(passthrough)}\0{self.transition_tree}\0{self.pheromone_tree}"
        return sha256(payload.encode("utf-8")).hexdigest()

    @property
    def total_nodes(self) -> int:
        return self.transition_nodes + self.pheromone_nodes

    def clone(self) -> RMTGPIndividual:
        return deepcopy(self)


def _clear_evaluation_metadata(individual: RMTGPIndividual) -> None:
    """树发生遗传变更后，移除只属于 parent genotype 的评价证据。"""

    for name in (
        "fitness_breakdown",
        "racing_screen_breakdown",
        "racing_screen_score",
        "racing_high_breakdown",
        "racing_high_score",
        "racing_fidelity_tier",
    ):
        individual.metadata.pop(name, None)


def is_baseline_individual(individual: RMTGPIndividual) -> bool:
    """个体是否为精确旁路 GP 的原始 ACO 哨兵。"""

    return bool(individual.metadata.get("baseline_passthrough", False))


def _random_tree(
    pset: gp.PrimitiveSetTyped,
    return_type: type,
    config: GPConfig,
) -> gp.PrimitiveTree:
    expression = gp.genHalfAndHalf(
        pset=pset,
        min_=config.initial_min_depth,
        max_=config.initial_max_depth,
        type_=return_type,
    )
    return gp.PrimitiveTree(expression)


def make_individual(
    transition_pset: gp.PrimitiveSetTyped,
    pheromone_pset: gp.PrimitiveSetTyped,
    config: GPConfig,
    *,
    mode: str = "joint",
) -> RMTGPIndividual:
    """按指定初始模式构造双树个体。"""

    zero_tr = constant_zero_tree(transition_pset, "ZERO_TR")
    zero_ph = constant_zero_tree(pheromone_pset, "ZERO_PH")

    def random_tr() -> gp.PrimitiveTree:
        return _random_tree(transition_pset, TrField, config)

    def random_ph() -> gp.PrimitiveTree:
        return _random_tree(pheromone_pset, PhField, config)

    if mode == "baseline":
        individual = RMTGPIndividual(zero_tr, zero_ph)
        individual.metadata["baseline_passthrough"] = True
        return individual
    # 初始生成也必须满足全局容量约束，不能只依赖后续 crossover/mutation
    # 的回退。深度上限很小，正常配置下会很快接受。
    for _ in range(10_000):
        if mode == "transition":
            individual = RMTGPIndividual(random_tr(), zero_ph)
        elif mode == "pheromone":
            individual = RMTGPIndividual(zero_tr, random_ph())
        elif mode == "joint":
            individual = RMTGPIndividual(random_tr(), random_ph())
        else:
            raise ValueError(f"未知个体初始化模式: {mode}")
        if valid_size(individual, config):
            return individual
    raise RuntimeError("无法在 GP 深度/节点约束内初始化合法个体")


def initialise_population(
    config: GPConfig,
    transition_pset: gp.PrimitiveSetTyped | None = None,
    pheromone_pset: gp.PrimitiveSetTyped | None = None,
) -> tuple[
    list[RMTGPIndividual],
    gp.PrimitiveSetTyped,
    gp.PrimitiveSetTyped,
]:
    """建立初始种群；安全协议下只保留一个不可变 baseline anchor。"""

    if transition_pset is None or pheromone_pset is None:
        transition_pset, pheromone_pset = create_primitive_sets(
            transition_profile=config.transition_profile,
            function_profile=config.function_profile,
            transition_terminals=config.transition_terminals,
            pheromone_terminals=config.pheromone_terminals,
        )
    if config.train_transition and config.train_pheromone:
        baseline_count = (
            1
            if config.baseline_anchor
            else round(config.population_size * 0.10)
        )
        counts = {
            "baseline": baseline_count,
            "transition": round(config.population_size * 0.20),
            "pheromone": round(config.population_size * 0.20),
        }
        counts["joint"] = config.population_size - sum(counts.values())
    elif config.train_transition:
        baseline_count = (
            1
            if config.baseline_anchor
            else round(config.population_size * 0.10)
        )
        counts = {
            "baseline": baseline_count,
            "transition": config.population_size - baseline_count,
        }
    else:
        baseline_count = (
            1
            if config.baseline_anchor
            else round(config.population_size * 0.10)
        )
        counts = {
            "baseline": baseline_count,
            "pheromone": config.population_size - baseline_count,
        }

    population = [
        make_individual(
            transition_pset,
            pheromone_pset,
            config,
            mode=mode,
        )
        for mode, count in counts.items()
        for _ in range(count)
    ]
    random.shuffle(population)
    return population, transition_pset, pheromone_pset


def valid_size(individual: RMTGPIndividual, config: GPConfig) -> bool:
    """检查双树的深度、节点数和常数有限性。"""

    return individual.total_nodes <= config.max_total_nodes and all(
        tree.height <= config.max_depth
        and len(tree) <= config.max_nodes_per_tree
        and expression_is_finite(tree)
        for tree in individual
    )


def _active_role(config: GPConfig) -> int:
    """按冻结的角色概率选择要修改的树。"""

    if config.train_transition and config.train_pheromone:
        return 1 if random.random() < config.pheromone_role_probability else 0
    return 0 if config.train_transition else 1


def mate_role_preserving(
    first: RMTGPIndividual,
    second: RMTGPIndividual,
    config: GPConfig,
) -> tuple[RMTGPIndividual, RMTGPIndividual]:
    """只在两个 parent 的同角色树之间执行 typed crossover。"""

    backup_first = first.clone()
    backup_second = second.clone()
    role = _active_role(config)
    gp.cxOnePoint(first[role], second[role])
    first_changed = valid_size(first, config)
    second_changed = valid_size(second, config)
    if not first_changed:
        first = backup_first
    if not second_changed:
        second = backup_second
    if first_changed:
        first.metadata["baseline_passthrough"] = False
        _clear_evaluation_metadata(first)
    if second_changed:
        second.metadata["baseline_passthrough"] = False
        _clear_evaluation_metadata(second)
    if first.fitness.valid:
        del first.fitness.values
    if second.fitness.valid:
        del second.fitness.values
    return first, second


def mutate_role_preserving(
    individual: RMTGPIndividual,
    transition_pset: gp.PrimitiveSetTyped,
    pheromone_pset: gp.PrimitiveSetTyped,
    config: GPConfig,
) -> tuple[RMTGPIndividual]:
    """在随机角色内执行 subtree、point 或 ERC mutation。"""

    backup = individual.clone()
    role = _active_role(config)
    tree = individual[role]
    pset = transition_pset if role == 0 else pheromone_pset
    draw = random.random()
    if draw < 0.50:
        expression = partial(gp.genFull, min_=0, max_=2)
        gp.mutUniform(tree, expr=expression, pset=pset)
    elif draw < 0.80:
        gp.mutNodeReplacement(tree, pset=pset)
    else:
        perturb_erc(tree)

    changed = valid_size(individual, config)
    if not changed:
        individual = backup
    else:
        individual.metadata["baseline_passthrough"] = False
        _clear_evaluation_metadata(individual)
    if individual.fitness.valid:
        del individual.fitness.values
    return (individual,)


def _fresh_active_individual(
    template: RMTGPIndividual,
    transition_pset: gp.PrimitiveSetTyped,
    pheromone_pset: gp.PrimitiveSetTyped,
    config: GPConfig,
) -> RMTGPIndividual:
    """重新生成活动树，同时原样保留冻结角色。"""

    mode = (
        "joint"
        if config.train_transition and config.train_pheromone
        else ("transition" if config.train_transition else "pheromone")
    )
    for _ in range(10_000):
        fresh = make_individual(
            transition_pset,
            pheromone_pset,
            config,
            mode=mode,
        )
        if not config.train_transition:
            fresh.transition_tree = deepcopy(template.transition_tree)
        if not config.train_pheromone:
            fresh.pheromone_tree = deepcopy(template.pheromone_tree)
        if valid_size(fresh, config):
            fresh.metadata["baseline_passthrough"] = False
            _clear_evaluation_metadata(fresh)
            if fresh.fitness.valid:
                del fresh.fitness.values
            return fresh
    raise RuntimeError("无法在结构副本约束下生成合法的新个体")


def _enforce_structural_copy_limit(
    population: list[RMTGPIndividual],
    transition_pset: gp.PrimitiveSetTyped,
    pheromone_pset: gp.PrimitiveSetTyped,
    config: GPConfig,
) -> list[RMTGPIndividual]:
    """限制完整双树 genotype 的副本数，避免 racing 后种群塌缩。"""

    limit = config.max_structural_copies
    if limit <= 0:
        return population
    counts: dict[str, int] = {}
    result: list[RMTGPIndividual] = []
    for original in population:
        candidate = original
        attempts = 0
        while counts.get(candidate.structural_hash, 0) >= limit:
            if attempts < 32:
                candidate = candidate.clone()
                (candidate,) = mutate_role_preserving(
                    candidate,
                    transition_pset,
                    pheromone_pset,
                    config,
                )
            else:
                candidate = _fresh_active_individual(
                    original,
                    transition_pset,
                    pheromone_pset,
                    config,
                )
            attempts += 1
            if attempts > 10_000:
                raise RuntimeError("无法满足 max_structural_copies 约束")
        key = candidate.structural_hash
        counts[key] = counts.get(key, 0) + 1
        result.append(candidate)
    return result


def evolve_generation(
    population: list[RMTGPIndividual],
    transition_pset: gp.PrimitiveSetTyped,
    pheromone_pset: gp.PrimitiveSetTyped,
    config: GPConfig,
    *,
    selection_key: Callable[[RMTGPIndividual], tuple[object, ...]] | None = None,
) -> list[RMTGPIndividual]:
    """根据预注册概率产生下一代，并可保留不可繁殖的 baseline anchor。"""

    if config.baseline_anchor:
        anchors = [item for item in population if is_baseline_individual(item)]
        if len(anchors) != 1:
            raise ValueError(
                "baseline_anchor 协议要求每一代恰好包含一个 baseline 哨兵"
            )
        anchor = anchors[0].clone()
        breeding_pool = [
            item for item in population if not is_baseline_individual(item)
        ]
        target_size = config.population_size - 1
    else:
        anchor = None
        breeding_pool = population
        target_size = config.population_size
    if not breeding_pool:
        raise ValueError("可繁殖的 GP 个体集合不得为空")

    rank_key = (
        selection_key
        if selection_key is not None
        else lambda item: (item.fitness.values, item.total_nodes)
    )
    ranked = sorted(breeding_pool, key=rank_key)

    def tournament(count: int) -> list[RMTGPIndividual]:
        """按可选多保真次序执行有放回 tournament。"""

        return [
            min(
                (
                    random.choice(breeding_pool)
                    for _ in range(config.tournament_size)
                ),
                key=rank_key,
            )
            for _ in range(count)
        ]
    elite_count = min(config.elite_size, target_size)
    elites = [item.clone() for item in ranked[:elite_count]]
    offspring: list[RMTGPIndividual] = []

    while len(elites) + len(offspring) < target_size:
        draw = random.random()
        if draw < config.crossover_probability:
            parents = tournament(2)
            child_a, child_b = (parents[0].clone(), parents[1].clone())
            child_a, child_b = mate_role_preserving(child_a, child_b, config)
            offspring.extend([child_a, child_b])
        elif draw < config.crossover_probability + config.mutation_probability:
            parent = tournament(1)[0]
            child = parent.clone()
            (child,) = mutate_role_preserving(
                child,
                transition_pset,
                pheromone_pset,
                config,
            )
            offspring.append(child)
        else:
            parent = tournament(1)[0]
            offspring.append(parent.clone())

    result = _enforce_structural_copy_limit(
        (elites + offspring)[:target_size],
        transition_pset,
        pheromone_pset,
        config,
    )
    if anchor is not None:
        result.append(anchor)
    return result


def evaluate_invalid(
    population: Iterable[RMTGPIndividual],
    evaluator: Callable[[RMTGPIndividual], float],
) -> int:
    """评估无 fitness 个体，并按 structural hash 做代内去重。"""

    cache: dict[str, float] = {}
    count = 0
    for individual in population:
        if individual.fitness.valid:
            continue
        key = individual.structural_hash
        if key not in cache:
            cache[key] = float(evaluator(individual))
            count += 1
        individual.fitness.values = (cache[key],)
    return count


def compile_individual(
    individual: RMTGPIndividual,
) -> tuple[TensorProgram | None, TensorProgram | None]:
    """编译双树；baseline sentinel 直接旁路 GP，适用于 replacement 消融。"""

    if bool(individual.metadata.get("baseline_passthrough", False)):
        return None, None
    return (
        compile_tree(individual.transition_tree, role="transition"),
        compile_tree(individual.pheromone_tree, role="pheromone"),
    )
