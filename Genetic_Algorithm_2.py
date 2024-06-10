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
def tournament_selection(population, fitnesses, num_selected):
    selected_indices = np.random.choice(len(population), num_selected, replace=False)
    selected_fitnesses = [fitnesses[i] for i in selected_indices]
    best_indices = np.argsort(selected_fitnesses)[-2:]  # Get indices of the two best individuals
    best_parents = [population[selected_indices[i]] for i in best_indices]
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


def genetic_algorithm(data, thresholds, classifiers, regressors, population_size, generations, crossover_prob, mutation_prob, tournament_size=3):
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
    optimizer = Optimize_eps(data, thresholds, classifiers, regressors)
    num_thresholds = len(thresholds)
    num_chromosomes = len(thresholds)

    # 1. First generation population is initialized
    population = initialize_population(num_thresholds, num_chromosomes, population_size)

    for generation in range(generations):
        # 2. Evaluate the fitness of each chromosome (=[sharpe, ret, vol])
        fitnesses = [optimizer.evaluate(chromosome) for chromosome in population]
        fitness_values = np.array([f[0] for f in fitnesses])

        # 3. Form new population taking into account different selection method: elitism, uniform crossover and uniform mutation
        new_population = []

        # 3.1. Elitism: carry the best chromosome to the next generation
        best_index = np.argmax(fitness_values)
        new_population.append(population[best_index])

        used_indices = {best_index}

        while len(new_population) < population_size:
            # Select a random pool of n individuals
            pool_indices = set()
            while len(pool_indices) < tournament_size:
                index = random.choice(range(population_size))
                if index not in used_indices:
                    pool_indices.add(index)
            
            pool_indices = list(pool_indices)
            pool = [population[i] for i in pool_indices]
            pool_fitnesses = [fitness_values[i] for i in pool_indices]

            # Select the best two individuals from the pool
            parents = tournament_selection(pool, pool_fitnesses, tournament_size)
            parent1, parent2 = parents

            used_indices.update(pool_indices) #each individual can be a parent only once

            # 3.2. Uniform Crossover: generate random number between 0 and 1 to decide if this chromosome will be recombined or replicate their parents
            if random.random() < crossover_prob:
                # for uniform crossover, each gene has the same 0.5 probability of being swapped
                child1, child2 = uniform_crossover(parent1, parent2)
            else:
                child1, child2 = parent1[:], parent2[:]

            # 3.3. Uniform Mutation: Iterate over each gene and mutate if random number is lower than mutation probability
        
            child1 = uniform_mutation(child1, mutation_prob)
            child2 = uniform_mutation(child2, mutation_prob)

            new_population.extend([child1, child2])

        # New population is formed
        population = new_population[:population_size]

        fitnesses = [optimizer.evaluate(chromosome) for chromosome in population]
        fitness_values = np.array([f[0] for f in fitnesses])
        best_fitness = max(fitness_values)
        print(f"Generation {generation}: Best Fitness = {best_fitness}")

    best_chromosome = population[np.argmax(fitness_values)]
    return best_chromosome