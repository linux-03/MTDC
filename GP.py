import numpy as np
import random

class GP_optimizer:
    def __init__(self, fitness_func, thresholds_count, population_size=500, generation_size=50, tournament_size=7, crossover_prob=0.90,
                 mutation_prob=0.1, elitism=0.1) -> None:
        
        self.fitness_func = fitness_func
        self.th_count = thresholds_count
        self.pop = []
        self.fintess = []
        # initialize population

        # first base chromosomes
        for i in range(thresholds_count):
            temp = []
            for j in range(thresholds_count):
                if i == j:
                    temp.append(1)
                else:
                    temp.append(0)
            
            self.pop.append(temp)
        
        # now the rest of the population
        for i in range(thresholds_count, population_size):
            temp = []
            for j in range(thresholds_count):
                temp.append(random.random())
        

    def eval_fitness(self):
        for j in self.pop:
            j.evaluate


        

        
