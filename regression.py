import copy
import math
import random

max_depth = 3
pop_size = 500
elitism_ratio = 0.1
generation_size = 35
mutation_rate = 0.02
crossover_rate = 0.98
tournament_size = 3
elite_size = int(elitism_ratio * pop_size)

terminals = ['DCl', 'ERC']
arity1 = ['sin', 'cos', 'log']
arity2 = ['+', '-', '/', '*']
arity = {fun: 1 for fun in arity1} | {fun: 2 for fun in arity2}
functions = [*arity1, *arity2]


##############################################################################
#####################       Individuals generation       #####################
##############################################################################

class Node:
    def __init__(self, val=None, children=None):
        self.val = val
        self.children = [] if children is None else children


def ERC():
    return random.uniform(-10, 10)


def generate_grow(depth):
    if depth <= 0:
        terminal = random.choice(terminals)
        return Node(val=(terminal if terminal != 'ERC' else ERC()))

    val = random.choice([*terminals, *functions])
    if val in terminals:
        return Node(val=(val if val != 'ERC' else ERC()))
    else:
        return Node(val=val, children=[generate_grow(depth - 1) for _ in range(arity[val])])


def generate_full(depth):
    if depth <= 0:
        terminal = random.choice(terminals)
        return Node(val=(terminal if terminal != 'ERC' else ERC()))
    
    fun = random.choice(functions)
    return Node(val=fun, children=[generate_full(depth - 1) for _ in range(arity[fun])])


def generate_half_half(depth):
    return generate_grow(depth) if random.random() < 0.5 else generate_full(depth)


##############################################################################
#####################              Fitness               #####################
##############################################################################

def eval_tree(node, DCl):
    if not len(node.children):
        if node.val == 'DCl':
            return DCl
        elif isinstance(node.val, (int, float)):
            return node.val
    elif len(node.children) == 1:
        value = eval_tree(node.children[0], DCl)
        if node.val == 'sin':
            return math.sin(value)
        elif node.val == 'cos':
            return math.cos(value)
        elif node.val == 'log':
            return math.log(value) if value > 0 else 0
    elif len(node.children) == 2:
        value0 = eval_tree(node.children[0], DCl)
        value1 = eval_tree(node.children[1], DCl)
        if node.val == '+':
            return value0 + value1
        elif node.val == '-':
            return value0 - value1
        elif node.val == '/':
            return value0 / value1 if value1 != 0 else 1
        elif node.val == '*':
            return value0 * value1

    return 0


def fitness(data, individual):
    #print(math.sqrt(sum([abs((1 - eval_tree(individual, DCl)/OSl)*100) for DCl, OSl in data]) / len(data)))
    return math.sqrt(sum([abs((1 - eval_tree(individual, DCl)/OSl)*100) for DCl, OSl in data]) / len(data))


def fittest(sample):
    loss, best_individual = min(sample, key=lambda x: x[0])

    return loss, best_individual


##############################################################################
#####################             Evolution              #####################
##############################################################################

def mutate(tree, depth):
    if random.random() < 0.5:
        # generate completely new tree
        return generate_half_half(depth)
    else:
        # only mutate childrens
        tree.children = [mutate(child, depth - 1) for child in tree.children]

        return tree


def offspring(parent1, parent2, depth):
    if (random.random() < 0.5 and depth > 0) or not len(parent1.children) or not len(parent2.children):
        # swap
        return parent2, parent1

    random_child1 = random.choice(range(len(parent1.children)))
    random_child2 = random.choice(range(len(parent2.children)))

    offspring1, offspring2 = offspring(parent1.children[random_child1], parent2.children[random_child2], depth+1)

    parent1.children[random_child1] = offspring1
    parent2.children[random_child2] = offspring2

    return parent1, parent2


def crossover(parent1, parent2):
    parent1 = copy.deepcopy(parent1)
    parent2 = copy.deepcopy(parent2)

    offspring1, offspring2 = offspring(parent1, parent2, depth=0)

    return offspring1 if random.random() < 0.5 else offspring2


##############################################################################
#####################               SRGP                 #####################
##############################################################################

def SRGP(data):
    pop = [generate_half_half(max_depth) for _ in range(pop_size)]
    fitnesses = [fitness(data, individual) for individual in pop]

    for generation in range(generation_size):
        newpop = [individual for _, individual in sorted(zip(fitnesses, pop), key=lambda x: x[0])]

        for i in range(elite_size, pop_size):
            if random.random() < crossover_rate:
                _, parent1 = fittest(random.sample(list(zip(fitnesses, pop)), tournament_size))
                _, parent2 = fittest(random.sample(list(zip(fitnesses, pop)), tournament_size))
                newpop[i] = crossover(parent1, parent2)

            if random.random() < mutation_rate:
                newpop[i] = mutate(newpop[i], max_depth)

        pop = newpop
        fitnesses = [fitness(data, individual) for individual in pop]
    
    loss, best_individual = fittest(list(zip(fitnesses, pop)))

    return loss, best_individual               
