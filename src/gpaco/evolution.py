"""DEAP 外层同步进化；全部逻辑任务完成后才选择下一代。"""

import random
from copy import deepcopy
from functools import partial

import numpy as np
from deap import gp

from .language import ProgramSpec, parse_tree, primitive_set


def valid(tree):
    return len(tree) <= 31 and tree.height <= 5


def initial_population(size):
    pset = primitive_set()
    population = [parse_tree("ZERO") for _ in range(round(size * 0.10))]
    while len(population) < size:
        tree = gp.PrimitiveTree(gp.genHalfAndHalf(pset=pset, min_=2, max_=4))
        if valid(tree):
            population.append(tree)
    random.shuffle(population)
    return population


def mutation(tree, pset):
    before = deepcopy(tree)
    draw = random.random()
    if draw < 0.5:
        gp.mutUniform(tree, expr=partial(gp.genFull, min_=0, max_=2), pset=pset)
    elif draw < 0.8:
        gp.mutNodeReplacement(tree, pset=pset)
    else:
        choices = [
            i
            for i, node in enumerate(tree)
            if not node.arity and isinstance(node.value, (int, float))
        ]
        if choices:
            index = random.choice(choices)
            node = tree[index]
            value = float(np.float32(np.clip(float(node.value) + random.gauss(0, 0.1), -1, 1)))
            tree[index] = gp.Terminal(value, False, node.ret)
    return tree if valid(tree) else before


def next_population(population, fitness):
    size = len(population)
    ranking = sorted(
        range(size),
        key=lambda i: (
            float(fitness[i]),
            len(population[i]),
            ProgramSpec.from_tree(population[i]).semantic_hash,
        ),
    )
    pset = primitive_set()
    result = [deepcopy(population[i]) for i in ranking[: max(1, size // 10)]]

    def tournament():
        candidates = random.choices(range(size), k=4)
        index = min(candidates, key=lambda i: (float(fitness[i]), len(population[i]), i))
        return deepcopy(population[index])

    while len(result) < size:
        draw = random.random()
        if draw < 0.8:
            a, b = tournament(), tournament()
            backup_a, backup_b = deepcopy(a), deepcopy(b)
            gp.cxOnePoint(a, b)
            result.extend([a if valid(a) else backup_a, b if valid(b) else backup_b])
        elif draw < 0.95:
            result.append(mutation(tournament(), pset))
        else:
            result.append(tournament())
    return result[:size]
