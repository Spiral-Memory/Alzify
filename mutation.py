from deap import base, creator, tools
import random

# Create the fitness and individual classes (assuming a maximization problem)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Define the mutation operator
toolbox = base.Toolbox()
toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.1, indpb=1)

# Example individual
individual = [0.5, 0.3, 0.8, 0.2]
print("Original Individual:", individual)

# Apply mutation to the individual
mutated_individual = toolbox.mutate(individual)


print("Mutated Individual:", mutated_individual)
