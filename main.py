# This is an evolutionary algorithm for the knapsack problem
# link to the original video
#   https://www.youtube.com/watch?v=nhT56blfRpE&t=381s

# main functions needed:
"""
1) genetic representation of a solution

2) a function to generate new solutions

3) fitness function

4) selection function

5) mutation function

6) crossover function
"""

from collections import namedtuple
from typing import List, Callable, Tuple
from random import choices
import random as r
import time
from functools import partial

Genome = List[int]
Population = List[Genome]

# TemplateFunc = Callable[[args], output]
FitnessFunc = Callable[[Genome], int] # rates the genome's fitness
PopulateFunc = Callable[[], Population] # produces new populations (from nothing)
SelectionFunc = Callable[[Population, FitnessFunc], Tuple[Genome, Genome]] # uses [P, FF] to decide the parents for next generation
CrossoverFunc = Callable[[Genome, Genome], Tuple[Genome, Genome]] # takes 2 genomes, returns 2 genomes
MutationFunc = Callable[[Genome], Genome] # 1 genome, returns 1 (sometimes modified) genome
Thing = namedtuple('Thing', ['name', 'value', 'weight'])

things = [
    Thing('Laptop', 500, 2200),
    Thing('Headphones', 150, 160),
    Thing('Coffee Mug', 60, 350),
    Thing('Notepad', 40, 333),
    Thing('Water Bottle', 30, 192),
]

# Generate a random list of 0 or 1, with len=k
def generate_genome(length: int) -> Genome:
    return choices([0, 1], k=length)


# To generate a population, render multiple genomes 
def generate_population(size: int, genome_length: int) -> Population:
    return [generate_genome(genome_length) for _ in range(size)]


def fitness(genome: Genome, things: List[Thing], weight_limit: int) -> int:
    if len(genome) != len(things):
        raise ValueError('genome and things must be the same length')
    
    weight = 0
    value = 0
    
    for i, thing in enumerate(things):
        if genome[i] == 1:
            weight += thing.weight
            value += thing.value
            
            # if we exceed our weight limit, return a '0' fitness and abort process
            if weight > weight_limit:
                return 0
            
    return value


def selection_pair(population: Population, fitness_func: FitnessFunc) -> Population:
    return choices(
        population=population,
        weights=[fitness_func(genome) for genome in population],
        k = 2
    )


def single_point_crossover(a: Genome, b: Genome) -> Tuple[Genome, Genome]:
    if len(a) != len(b):
        raise ValueError('Genomes a and b must be of same length')
    
    length = len(a)
    if length < 2:
        return a, b
    
    p = r.randint(1, length - 1)
    return a[0:p] + b[p:], b[0:p] + a[p:]


def mutation(genome: Genome, num: int = 1, probability: float = 0.5) -> Genome:
    for _ in range(num):
        index = r.randrange(len(genome))
        genome[index] = genome[index] if r.random() > probability else abs(genome[index] - 1)
    return genome


def run_evolution(
    populate_func: PopulateFunc,
    fitness_func: FitnessFunc,
    fitness_limit: int,
    selection_func: SelectionFunc = selection_pair,
    crossover_func: CrossoverFunc = single_point_crossover,
    mutation_func: MutationFunc = mutation,
    generation_limit: int = 100
) -> Tuple[Population, int]:
    population = populate_func()
    
    for i in range(generation_limit):
        population = sorted(
            population,
            key=lambda genome: fitness_func(genome),
            reverse=True
        )
        
        if fitness_func(population[0]) >= fitness_limit:
            break
        
        next_generation = population[0:2]
        
        for j in range(int(len(population) / 2) - 1):
            parents = selection_func(population, fitness_func)
            offspring_a, offspring_b = crossover_func(parents[0], parents[1])
            offspring_a = mutation_func(offspring_a)
            offspring_b = mutation_func(offspring_b)
            next_generation += [offspring_a, offspring_b]
        
        population = next_generation
    
    population = sorted(
        population,
        key=lambda genome: fitness_func(genome),
        reverse=True
    )
    
    return population, i

start = time.time()
population, generations = run_evolution(
    populate_func=partial(
        generate_population, size=10, genome_length=len(things)
    ),
    fitness_func=partial(
        fitness, things=things, weight_limit=3000
    ),
    fitness_limit=740,
    generation_limit=100
)
end = time.time()

def genome_to_things(genome: Genome, things: List[Thing]) -> List[Thing]:
    result = []
    for i, thing in enumerate(things):
        if genome[i] == 1:
            result += [thing.name]
    
    return result


print(f'\n\nnumber of generations: {generations}')
print(f'time: {end - start:.6f}s')
print(f'best solution: {genome_to_things(population[0], things)}\n\n')
