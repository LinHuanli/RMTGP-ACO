"""DEAP Strongly Typed 双树个体与遗传算子。"""

from __future__ import annotations

from copy import deepcopy
from functools import partial
from hashlib import sha256
import random
from typing import Callable, Iterable

from deap import base, gp, tools

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

    @property
    def structural_hash(self) -> str:
        """对两棵树表达式计算可用于 memoization 的哈希。"""

        passthrough = bool(self.metadata.get("baseline_passthrough", False))
        payload = f"{int(passthrough)}\0{self.transition_tree}\0{self.pheromone_tree}"
        return sha256(payload.encode("utf-8")).hexdigest()

    @property
    def total_nodes(self) -> int:
        return len(self.transition_tree) + len(self.pheromone_tree)

    def clone(self) -> "RMTGPIndividual":
        return deepcopy(self)


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
    random_tr = lambda: _random_tree(transition_pset, TrField, config)
    random_ph = lambda: _random_tree(pheromone_pset, PhField, config)

    if mode == "baseline":
        individual = RMTGPIndividual(zero_tr, zero_ph)
        individual.metadata["baseline_passthrough"] = True
        return individual
    if mode == "transition":
        return RMTGPIndividual(random_tr(), zero_ph)
    if mode == "pheromone":
        return RMTGPIndividual(zero_tr, random_ph())
    if mode == "joint":
        return RMTGPIndividual(random_tr(), random_ph())
    raise ValueError(f"未知个体初始化模式: {mode}")


def initialise_population(
    config: GPConfig,
    transition_pset: gp.PrimitiveSetTyped | None = None,
    pheromone_pset: gp.PrimitiveSetTyped | None = None,
) -> tuple[
    list[RMTGPIndividual],
    gp.PrimitiveSetTyped,
    gp.PrimitiveSetTyped,
]:
    """按 10/20/20/50 比例建立初始种群。"""

    if transition_pset is None or pheromone_pset is None:
        transition_pset, pheromone_pset = create_primitive_sets(
            transition_profile=config.transition_profile,
            function_profile=config.function_profile,
            transition_terminals=config.transition_terminals,
            pheromone_terminals=config.pheromone_terminals,
        )
    if config.train_transition and config.train_pheromone:
        counts = {
            "baseline": round(config.population_size * 0.10),
            "transition": round(config.population_size * 0.20),
            "pheromone": round(config.population_size * 0.20),
        }
        counts["joint"] = config.population_size - sum(counts.values())
    elif config.train_transition:
        counts = {
            "baseline": round(config.population_size * 0.10),
            "transition": config.population_size - round(config.population_size * 0.10),
        }
    else:
        counts = {
            "baseline": round(config.population_size * 0.10),
            "pheromone": config.population_size - round(config.population_size * 0.10),
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

    return all(
        tree.height <= config.max_depth
        and len(tree) <= config.max_nodes_per_tree
        and expression_is_finite(tree)
        for tree in individual
    )


def mate_role_preserving(
    first: RMTGPIndividual,
    second: RMTGPIndividual,
    config: GPConfig,
) -> tuple[RMTGPIndividual, RMTGPIndividual]:
    """只在两个 parent 的同角色树之间执行 typed crossover。"""

    backup_first = first.clone()
    backup_second = second.clone()
    active_roles = [
        role
        for role, enabled in enumerate(
            (config.train_transition, config.train_pheromone)
        )
        if enabled
    ]
    role = random.choice(active_roles)
    gp.cxOnePoint(first[role], second[role])
    first_changed = valid_size(first, config)
    second_changed = valid_size(second, config)
    if not first_changed:
        first = backup_first
    if not second_changed:
        second = backup_second
    if first_changed:
        first.metadata["baseline_passthrough"] = False
    if second_changed:
        second.metadata["baseline_passthrough"] = False
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
    active_roles = [
        role
        for role, enabled in enumerate(
            (config.train_transition, config.train_pheromone)
        )
        if enabled
    ]
    role = random.choice(active_roles)
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
    if individual.fitness.valid:
        del individual.fitness.values
    return (individual,)


def evolve_generation(
    population: list[RMTGPIndividual],
    transition_pset: gp.PrimitiveSetTyped,
    pheromone_pset: gp.PrimitiveSetTyped,
    config: GPConfig,
) -> list[RMTGPIndividual]:
    """根据预注册概率产生下一代，保留显式 elites。"""

    ranked = sorted(population, key=lambda item: (item.fitness.values, item.total_nodes))
    elites = [item.clone() for item in ranked[: config.elite_size]]
    offspring: list[RMTGPIndividual] = []

    while len(elites) + len(offspring) < config.population_size:
        draw = random.random()
        if draw < config.crossover_probability:
            parents = tools.selTournament(population, 2, tournsize=config.tournament_size)
            child_a, child_b = (parents[0].clone(), parents[1].clone())
            child_a, child_b = mate_role_preserving(child_a, child_b, config)
            offspring.extend([child_a, child_b])
        elif draw < config.crossover_probability + config.mutation_probability:
            parent = tools.selTournament(population, 1, tournsize=config.tournament_size)[0]
            child = parent.clone()
            (child,) = mutate_role_preserving(
                child,
                transition_pset,
                pheromone_pset,
                config,
            )
            offspring.append(child)
        else:
            parent = tools.selTournament(population, 1, tournsize=config.tournament_size)[0]
            offspring.append(parent.clone())

    return (elites + offspring)[: config.population_size]


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
