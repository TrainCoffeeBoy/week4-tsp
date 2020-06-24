#!/usr/bin/env python3

import sys
import math

from common_ga import format_solution, read_input

import solver_ga_opt

CHALLENGES = 7

def generate_solution():
    for challenge_number in range(CHALLENGES):
        cities = read_input('input_{}.csv'.format(challenge_number))
        print("challenge_number: {} -------------------------------------".format(challenge_number))
        N, dist = solver_ga_opt.get_N_dist(cities)
        solution = solver_ga_opt.geneticAlgorithm(population=cities, popSize=100, eliteSize=20, mutationRate=0.15, generations=500, N=N, dist=dist)
        with open('solution_yours_{}.csv'.format(challenge_number), 'w') as f:
            f.write(format_solution(solution) + '\n')


if __name__ == '__main__':
    generate_solution()