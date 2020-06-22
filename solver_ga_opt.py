import sys, numpy as np, random, operator, pandas as pd, copy, math

from common_ga import print_solution, read_input

class City:
    def __init__(self, i, x, y):
        self.index = i
        self.visited = False
        self.x = x
        self.y = y

    def distance(self, city):
        return np.sqrt((self.x - city.x) ** 2 + (self.y - city.y) ** 2)


class Fitness:
    def __init__(self, route):
        self.route = route
        self.distance = 0
        self.fitness = 0.0
        self.each_dist = []

    def routeDistance(self):
        if self.distance == 0:
            pathDistance = 0
            for i in range(len(self.route)):
                fromCity = self.route[i]
                toCity = None
                if i + 1 < len(self.route):
                    toCity = self.route[i + 1]
                else:
                    toCity = self.route[0]
                dist = fromCity.distance(toCity)
                pathDistance += dist
                self.each_dist.append(dist)
            self.distance = pathDistance
        return self.distance

    def routeFitness(self):
        if self.fitness == 0:
            self.fitness = 1 / float(self.routeDistance())
        return self.fitness

def distance(city1, city2):
    return math.sqrt((city1.x - city2.x) ** 2 + (city1.y - city2.y) ** 2)


def solve(cities):
    N = len(cities)

    dist = [[0] * N for i in range(N)]
    for i in range(N):
        for j in range(N):
            dist[i][j] = dist[j][i] = distance(cities[i], cities[j])

    unvisited_cities = set(cities)
    current_city = random.sample(unvisited_cities, 1)[0]
    unvisited_cities.remove(current_city)
    solution = [current_city]

    def distance_from_current_city(to):
        return dist[current_city.index][to.index]

    while unvisited_cities:
        next_city = min(unvisited_cities, key=distance_from_current_city)
        unvisited_cities.remove(next_city)
        solution.append(next_city)
        current_city = next_city

    solution = two_opt(solution, dist, N)
    return solution


def two_opt(solution, dist, N):
    while True:
        count = 0
        for i in range(N - 2):
            for j in range(i + 2, N):
                l1 = dist[solution[i].index][solution[i + 1].index]
                l2 = dist[solution[j].index][solution[(j + 1) % N].index]
                l3 = dist[solution[i].index][solution[j].index]
                l4 = dist[solution[i + 1].index][solution[(j + 1) % N].index]
                if l1 + l2 > l3 + l4:
                    new_solution = solution[i + 1:j + 1]
                    solution[i + 1:j + 1] = new_solution[::-1]
                    count += 1
        if count == 0:
            break

    return solution


def createRandomRoute(cityList):
    route = random.sample(cityList, len(cityList))
    return route


def initialPopulation(Popsize, cities):
    population = []
    nn_opt = int(Popsize * 0.2)
    for _ in range(nn_opt):
        population.append(solve(cities))
    for _ in range(Popsize - nn_opt):
        population.append(createRandomRoute(cities))
    return population


def rankRoutes(population):
    fitnessResults = {}
    for i in range(len(population)):
        fitnessResults[i] = Fitness(population[i]).routeFitness()
    return sorted(fitnessResults.items(), key = operator.itemgetter(1), reverse = True)


def selection(popRanked, eliteSize):
    selectionResults = []
    df = pd.DataFrame(np.array(popRanked), columns = ["index", "Fitness"])
    df["cum_sum"] = df.Fitness.cumsum()
    df["cum_perc"] = 100 * df.cum_sum / df.Fitness.sum()

    for i in range(eliteSize):
        selectionResults.append(popRanked[i][0])
    for i in range(len(popRanked) - eliteSize):
        pick = 100 * random.random()
        for i in range(len(popRanked)):
            if pick <= df.iat[i, 3]:
                selectionResults.append(popRanked[i][0])
                break
    return selectionResults


def matingPool(population, selectionResults):
    matingpool = []
    for i in range(len(selectionResults)):
        index = selectionResults[i]
        matingpool.append(population[index])
    return matingpool


def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []

    geneA = int(random.random() * len(parent1))
    geneB = int(random.random() * len(parent2))

    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])

    childP2 = [item for item in parent2 if item not in childP1]
    child = childP1 + childP2 
    return child

def breedPopulation(matingpool, eliteSize):
    children = []
    length = len(matingpool) - eliteSize
    pool = random.sample(matingpool, len(matingpool))

    for i in range(eliteSize):
        children.append(matingpool[i])

    for i in range(length):
        child = breed(pool[i], pool[len(matingpool) - i - 1])
        children.append(child)   
    return children


def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if (random.random() < mutationRate):
            swapWith = int(random.random() * len(individual))

            city1 = individual[swapped]
            city2 = individual[swapWith]

            individual[swapped] = city1
            individual[swapWith] = city2
    return individual

def mutatePopulation(population, mutationRate):
    mutatedPop = []

    for ind in range(len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)

    return mutatedPop

def nextGeneration(currentGen, eliteSize, mutationRate):
    popRanked = rankRoutes(currentGen)
    selectionResults = selection(popRanked, eliteSize)
    matingpool = matingPool(currentGen, selectionResults)
    children = breedPopulation(matingpool, eliteSize)
    nextGeneration = mutatePopulation(children, mutationRate)
    
    return nextGeneration

def geneticAlgorithm(population, popSize, eliteSize, mutationRate, generations):
    pop = initialPopulation(popSize, population)
    print("initial distance: {}".format(1 / rankRoutes(pop)[0][1]))

    for i in range(generations):
        pop = nextGeneration(pop, eliteSize, mutationRate)

    print("Final distance: {}".format(1 / rankRoutes(pop)[0][1]))
    bestRouteIndex = rankRoutes(pop)[0][0]
    bestRoute = pop[bestRouteIndex]
    bestRoute_index = []
    for city in bestRoute:
        bestRoute_index.append(city.index)

    return bestRoute_index


if __name__ == "__main__":
    assert len(sys.argv) > 1
    cities = read_input(sys.argv[1])
    solution = geneticAlgorithm(population=cities, popSize=10, eliteSize=2, mutationRate=0.10, generations=500)
    print(solution)