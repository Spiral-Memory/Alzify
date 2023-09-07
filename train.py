import torch
import torch.nn as nn
import torch.optim as optim
from model import build_model
from datasets_und_sm import create_data_loaders
import numpy as np
from deap import algorithms, base, creator, tools
from torch.utils.data import DataLoader

def evaluate(individual):

    model = build_model(pretrained=True, fine_tune=False).to(device)

    torch.manual_seed(42)

    # Set the hyperparameters from the individual
    lr = individual[0]  # Learning rate
    batch_size = individual[1]
    num_epochs = individual[2]

    lr = abs(lr)
    batch_size = int(abs(batch_size))
    num_epochs = int(abs(num_epochs))

    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_loader = DataLoader(oversampled_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    print('Training started ....\n')
    model.train()
    for epoch in range(num_epochs):
        for images,labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    

    model.eval()
    print('Validation started ....\n')
    # valid_running_loss = 0.0
    valid_running_correct = 0
    counter = 0
    with torch.no_grad():
        for images,labels in valid_loader:
            counter += 1
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            # valid_running_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()


    valid_acc = (valid_running_correct / len(valid_loader.dataset))
    
    # return valid_acc, valid_loss
    return valid_acc,


device = ('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Computation device: {device}\n")

# Create the fitness maximization problem
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# Define the toolbox for the genetic algorithm
toolbox = base.Toolbox()
toolbox.register("attr_lr", np.random.uniform, low=0.0009, high=0.01)
toolbox.register("attr_epochs", np.random.randint, low=5, high=22)
toolbox.register("attr_batch_size", np.random.randint, low=16, high=64)
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_lr, toolbox.attr_batch_size, toolbox.attr_epochs), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.01, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)


# Load the dataset
oversampled_dataset, test_dataset = create_data_loaders()

# Set the random seed for reproducibility
np.random.seed(42)

# Initialize the population
population_size = 20
population = toolbox.population(n=population_size)
print("Initial Population:", population)

# Run the genetic algorithm
num_generations = 5
for generation in range(num_generations):
    print(f"Generation {generation + 1}/{num_generations}")
    
    # Evaluate the population fitness
    fitnesses = map(toolbox.evaluate, population)
    for individual, fitness in zip(population, fitnesses):
        individual.fitness.values = fitness
        print("Individual:", individual, "Fitness:", fitness)
    
    # Select the next generation individuals
    offspring = toolbox.select(population, len(population))
    print("Offspring Selected :", offspring)
    
    # Apply crossover and mutation on the offspring
    offspring = algorithms.varAnd(offspring, toolbox, cxpb=0.5, mutpb=0.1)
    print("Offspring Created:", offspring)
    
    # Replace the population with the offspring
    population[:] = offspring
    print("Population Replaced:", population)

# Get the best individual and its fitness
best_individual = tools.selBest(population, k=1)
print("Best Individual:", best_individual)
