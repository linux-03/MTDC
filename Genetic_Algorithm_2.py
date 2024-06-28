import random
import numpy as np


def initialize_population(num_thresholds, num_chromosomes, population_size):
        population = []
        for i in range(num_thresholds):
            chromosome = [1.0 if j == i else 0.0 for j in range(num_chromosomes)]
            population.append(chromosome)
        for i in range(num_thresholds, population_size):
            chromosome = [random.uniform(0.0, 1.0) for _ in range(num_chromosomes)]
            population.append(chromosome)
        return population

# Function for tournament selection to select fittest individuals from mating pool to become parents
def tournament_selection(population, fitnesses):
    #selected_indices = np.random.choice(len(population), num_selected, replace=False)
    #selected_fitnesses = [fitnesses[i] for i in selected_indices]
    best_indices = np.argmax(fitnesses) # Get indices of the two best individuals
    best_parents = population[best_indices]
    return best_parents

# Function for uniform crossover
def uniform_crossover(parent1, parent2):
    child1, child2 = [], []
    for gene1, gene2 in zip(parent1, parent2):
        if random.random() < 0.5:
            child1.append(gene1)
            child2.append(gene2)
        else:
            child1.append(gene2)
            child2.append(gene1)
    return child1, child2

# Function for uniform mutation
def uniform_mutation(chromosome, mutation_prob):
    return [gene if random.random() > mutation_prob else random.uniform(0.0, 1.0) for gene in chromosome]


def genetic_algorithm(data, thresholds, backtest_strat  , population_size = 500, generations=35, crossover_prob=0.98, mutation_prob=0.02, tournament_size=3, elitism_ratio=0.1):
    '''
    Function to select the chromosome representing the most effective set of weights for the trading strategy thresholds, as evaluated by the backtesting process
    Returns: list containing the weights for the strategies

    Parameters
    ----------
    thresholds : list 
        List containing thresholds 
    classifiers: list 
        List containing the classifiers 
    regressors: list 
        List containing the regressors

    population_size : integer
        Number of initial options of best solutions
    generations : integer
        How often should we run the algorithm
    crossover_prob : float (range 0 - 1) 
        Probability to combine genes of the parents instead of only passing the parents genes, rather high 
    mutation_prob = float (range 0-1)
        Probability to randomly mutate, rather low
    tournament_size = integer
        Size of the mating pool from which the best individuals are selected to become parents. "Selection pressure"
    -----------
    '''
    optimizer = backtest_strat
    num_thresholds = len(thresholds)
    num_chromosomes = len(thresholds)

    # 1. First generation population is initialized
    population = initialize_population(num_thresholds, num_chromosomes, population_size)
    fitnesses = [optimizer.evaluate(chromosome) for chromosome in population]

    fitness_values = np.array(fitnesses)
    for generation in range(generations):
        # 2. Evaluate the fitness of each chromosome (=[sharpe, ret, vol])
        """
        fitnesses = [optimizer.evaluate(chromosome) for chromosome in population]

        fitness_values = np.array(fitnesses) 
        """

        # 3. Form new population taking into account different selection method: elitism, uniform crossover and uniform mutation
        #new_population = []

        # 3.1. Elitism: carry the best chromosome to the next generation
        best_index = sorted(fitness_values, reverse=True)[:int(elitism_ratio*population_size)]
        new_population = [population[i] for i in best_index]
        #used_indices = {best_index}

        while len(new_population) < population_size:
            # Select a random pool of n individuals
            pool_indices = set()
            #while len(pool_indices) < tournament_size:
            #    print('tournament')
            pool_indices_1 = random.sample(range(population_size), tournament_size)
            pool_indices_2 = random.sample(range(population_size), tournament_size)
            #    if index not in used_indices:
            #        pool_indices.add(index)
            
            #pool_indices = list(pool_indices)
            pool_1 = [population[i] for i in pool_indices_1]
            pool_2 = [population[i] for i in pool_indices_2]
            pool_fitnesses_1 = [fitness_values[i] for i in pool_indices_1]
            pool_fitnesses_2 = [fitness_values[i] for i in pool_indices_2]
            
            print('selection')
            # Select the best two individuals from the pool
            parent1 = tournament_selection(pool_1, pool_fitnesses_1)
            parent2 = tournament_selection(pool_2, pool_fitnesses_2)
            

            #used_indices.update(pool_indices) #each individual can be a parent only once
            print('crossover')
            # 3.2. Uniform Crossover: generate random number between 0 and 1 to decide if this chromosome will be recombined or replicate their parents
            if random.random() < crossover_prob:
                # for uniform crossover, each gene has the same 0.5 probability of being swapped
                child1, child2 = uniform_crossover(parent1, parent2)
            else:
                child1, child2 = parent1[:], parent2[:]
            print('mutation')
            # 3.3. Uniform Mutation: Iterate over each gene and mutate if random number is lower than mutation probability
        
            child1 = uniform_mutation(child1, mutation_prob)
            child2 = uniform_mutation(child2, mutation_prob)
            print('add new')
            new_population.extend([child1, child2])
            print('end')

        # New population is formed
        population = new_population[:population_size]

        fitnesses = [optimizer.evaluate(chromosome) for chromosome in population]
        fitness_values = np.array([f for f in fitnesses]) 
        
        best_fitness = max(fitness_values)
        print(f"Generation {generation}: Best Fitness = {best_fitness}")

    best_chromosome = population[np.argmax(fitness_values)]
    return best_chromosome
