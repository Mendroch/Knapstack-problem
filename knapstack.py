import random
from deap import base, creator, tools, algorithms
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Problem plecakowy
items = [
    (4, 12), (2, 1), (6, 4), (1, 2), (5, 10), (7, 7), (3, 8), (8, 15), (9, 25), (3, 5),
    (2, 2), (4, 6), (5, 18), (7, 20), (6, 16), (3, 10), (9, 24), (8, 22), (10, 30), (1, 1),
    (5, 14), (6, 8), (2, 3), (7, 12), (9, 20), (4, 9), (3, 7), (8, 18), (6, 15), (10, 28)
]
MAX_WEIGHT = 50

# Tworzenie typu osobnika i funkcji celu
creator.create("Fitness", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.Fitness)

toolbox = base.Toolbox()
toolbox.register("attr_bool", random.randint, 0, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=len(items))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Funkcja oceny
def evaluate(individual):
    total_weight = sum(ind * item[0] for ind, item in zip(individual, items))
    total_value = sum(ind * item[1] for ind, item in zip(individual, items))
    if total_weight > MAX_WEIGHT:
        return 0,  # Przekroczenie wagi powoduje dyskwalifikację
    return total_value,

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

# Parametry algorytmu
POP_SIZE = 100
N_GEN = 50
CXPB = 0.7
MUTPB = 0.2

# Przechowywanie wyników z wielu uruchomień
all_runs_best = []
all_runs_mean = []
all_runs_worst = []
best_individuals = []  # Genotyp najlepszego osobnika z każdego uruchomienia
best_values_over_runs = []  # Wartości plecaka
best_weights_over_runs = []  # Wagi plecaka

# Inicjalizacja zmiennych dla najlepszego osobnika globalnego
global_best_ind = None
global_best_value = -float('inf')
global_best_weight = None

for run in range(10):  # 10 uruchomień
    # Uruchomienie algorytmu
    pop = toolbox.population(n=POP_SIZE)
    best_fitness = []
    mean_fitness = []
    worst_fitness = []

    for gen in range(N_GEN):
        # Proces ewolucyjny (ten sam co wcześniej)
        fitnesses = list(map(toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring

        # Zapis wyników dla tego uruchomienia
        fits = [ind.fitness.values[0] for ind in pop]
        best_fitness.append(max(fits))
        worst_fitness.append(min(fits))
        mean_fitness.append(np.mean(fits))
        
    # Najlepszy osobnik z ostatniej generacji
    best_ind = tools.selBest(pop, 1)[0]
    best_value = best_ind.fitness.values[0]
    best_weight = sum(ind * item[0] for ind, item in zip(best_ind, items))

    best_individuals.append(list(best_ind))  # Przechowujemy genotyp najlepszego osobnika
    best_values_over_runs.append(best_value)
    best_weights_over_runs.append(best_weight)

    # Aktualizacja najlepszego osobnika globalnie
    if best_value > global_best_value:
        global_best_ind = list(best_ind)  # Genotyp najlepszego osobnika
        global_best_value = best_value    # Fitness najlepszego osobnika
        global_best_weight = best_weight  # Waga najlepszego osobnika

    # Zapis wyników z tego uruchomienia do ogólnej struktury
    all_runs_best.append(best_fitness)
    all_runs_mean.append(mean_fitness)
    all_runs_worst.append(worst_fitness)

# Obliczanie średnich i odchyleń standardowych dla każdej generacji
avg_best = np.mean(all_runs_best, axis=0)
std_best = np.std(all_runs_best, axis=0)

avg_mean = np.mean(all_runs_mean, axis=0)
std_mean = np.std(all_runs_mean, axis=0)

avg_worst = np.mean(all_runs_worst, axis=0)
std_worst = np.std(all_runs_worst, axis=0)

# Wykres fitnessu
generations = np.arange(1, N_GEN + 1)

plt.figure(figsize=(12, 6))
plt.plot(generations, avg_best, label="Średni najlepszy fitness", color='green', linewidth=2)
plt.fill_between(generations, avg_best - std_best, avg_best + std_best, color='green', alpha=0.3)

plt.plot(generations, avg_mean, label="Średni średni fitness", color='blue', linestyle='--', linewidth=2)
plt.fill_between(generations, avg_mean - std_mean, avg_mean + std_mean, color='blue', alpha=0.3)

plt.plot(generations, avg_worst, label="Średni najgorszy fitness", color='red', linestyle=':', linewidth=2)
plt.fill_between(generations, avg_worst - std_worst, avg_worst + std_worst, color='red', alpha=0.3)

# Dodatki do wykresu
plt.title("Średnie wartości fitness z odchyleniem standardowym (10 uruchomień)", fontsize=14)
plt.xlabel("Generacja", fontsize=12)
plt.ylabel("Fitness", fontsize=12)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# Mapa cieplna genotypów
plt.figure(figsize=(12, 6))
sns.heatmap(best_individuals, cmap="YlGnBu", cbar=True, linewidths=0.5)
plt.title("Wybrane przedmioty w najlepszych osobnikach (10 uruchomień)", fontsize=14)
plt.xlabel("Przedmiot", fontsize=12)
plt.ylabel("Numer uruchomienia", fontsize=12)
plt.show()

# Średnie i odchylenia dla wartości i wag
avg_value = np.mean(best_values_over_runs)
std_value = np.std(best_values_over_runs)

avg_weight = np.mean(best_weights_over_runs)
std_weight = np.std(best_weights_over_runs)

# Wykres wartości i wag
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
ax.bar(["Wartość", "Waga"], [avg_value, avg_weight], yerr=[std_value, std_weight], capsize=5, color=['blue', 'green'])
ax.set_title("Średnia wartość i waga plecaka z odchyleniem standardowym (10 uruchomień)", fontsize=14)
ax.set_ylabel("Średnia", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Wyświetlenie najlepszego osobnika globalnego
print("\nNajlepszy wynik z 10 uruchomień:")
print(f"Najlepszy osobnik: {global_best_ind}")
print(f"Wartość: {global_best_value}")
print(f"Waga: {global_best_weight}")
