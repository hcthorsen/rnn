#!/usr/bin/python3
# -*- coding: utf-8 -*-

# import libraries
import sys
import math
import random
import numpy as np
from matplotlib import pyplot as plt
import xlrd
import datetime
import time

# *****************************************************************************
# constants *******************************************************************
# *****************************************************************************

fit_var = {'PEN_HARD': 10000, 'PEN_MEDIUM': 10, 'PEN_SOFT': 1, 'DAY_MIN': 8,
           'DAY_MAX': 18, 'N_WEEKEND': 4, 'N_NIGHT': 2, 'N_LATE': 1,
           'TOLERANCE_HOUR': 10, 'TOLERANCE_NIGHT': 1, 'HOURS_DAY': 7.4,
           'HOURS_LATE': 13, 'HOURS_NIGHT': 18.5, 'CHIEF_PHYS_MIN': 1,
           'JUNIOR_PHYS_MAX': 1, 'EXTRA_WEEKDAYS': 5}

model_var = {'POP_SIZE': 100, 'GENERATIONS': 5000, 'MUTATE': True,
             'MUTATION_RATE': 10, 'PERMUTE': True, 'PERMUTATION_RATE': 50,
             'ELITISM': True, 'N_ELITISM': 10, 'LOCAL_SEARCH': True,
             'SEARCH_ALL': True, 'LOCAL_SEARCH_CRITERION': 100,
             'STOP_CRITERION': 200, 'TOURNAMENT': True, 'N_TOURNAMENT': 3,  
             'TWO_POINT_CROSSOVER': True, 'PRINT_GENERATION': 100, 
             'DEBUG': True, 'REMOVE_TWINS': False, 'PERMUTE_TWINS': False}

# *****************************************************************************
# import 'vagtønsker'**********************************************************
# *****************************************************************************

# Try to open the excel file and write an error if it fails
try:
    doc = xlrd.open_workbook('vagtplan_nov.xlsx').sheet_by_index(0)
except IOError as error:
    print("Can't open file, reason:", str(error))
    sys.exit(1)

# Extract names of employees
names = doc.col_values(0, 1, 27)

# Extract dates
dates = doc.row_values(0, 5, 38)

# Extract nightshift
nightshift = doc.col_values(2, 1, 27)

# Extract competence
competence = doc.col_values(4, 1, 27)

N = len(names)
M = len(dates)

# Preallocate memory, then extract data to matrix X
X = np.array(np.empty((N, M), dtype=(str, 5)))
for i in range(M):
    try:
        X[:, i] = np.array(doc.col_values(i+5, 1, N+1)).T
    except ValueError as error:
        print(str(error), i)

#N, M = X.shape

# Funtion that converts Excel datetime to Python datetime
def excel_time_to_string(excel_time, fmt='%d-%m-%Y'):
    dt = datetime.datetime(1899, 12, 30) + datetime.timedelta(days=excel_time)
    return dt.strftime(fmt)

# Function that converts Excel datetime to weekend 0/1
def excel_time_to_weekend(excel_time):
    dt = datetime.datetime(1899, 12, 30) + datetime.timedelta(days=excel_time)
    day = datetime.datetime.weekday(dt)
    if day == 5 or day == 6:
        weekend = 1
    else:
        weekend = 0
    return weekend

# Construct a list, that states if a date is in the weekend 
weekend = list(map(excel_time_to_weekend, dates))

# Convert excel dates to a proper format
dates = list(map(excel_time_to_string, dates))

# For simplicity employees enter and leave with a balance of 0 hours
balance_in = np.array(np.zeros((N), dtype=int))
balance_out = np.array(np.zeros((N), dtype=int))

# As a starting point employees have deliver hours corresponding to working
# every weekday
hours_to_deliver = np.array(np.ones((N), dtype=int)) \
                   * (fit_var['HOURS_DAY'] \
                   * (M-sum(weekend)))

# Employees are not supposed to deliver hours when they are on 
# vaccation, maternal leave etc.
for row in range(N):
    hours_to_deliver[row] -= (X[row, :] == 'Ba').sum() * fit_var['HOURS_DAY']
    hours_to_deliver[row] -= (X[row, :] == 'Fe').sum() * fit_var['HOURS_DAY']
    hours_to_deliver[row] -= (X[row,:]=='Tj').sum() * fit_var['HOURS_DAY']
    hours_to_deliver[row] -= (X[row,:]=='Om').sum() * fit_var['HOURS_DAY']

hours_to_deliver += balance_out - balance_in

# The array *X* is a 2D array with peoples requests. Each row correspond to a
# person, and each column correspond to a date. For further computation the
# array needs to be split up in to 3 arrays:
# 1 for day shift
# 1 for long shift
# 1 for night shift
# 3 dictionaries are used to split the array *X*. The 3 new arrays are
# assembled in a 3D array *requests*:
# Layer 0 concerns day shift
# layer 1 concerns late shift
# layer 2 concerns night shift

dict_day = {'Ba': -1, '-d': -1, '-l': 0, '-v': 0, '/F': -1, '+D': 1,
            '+I': 1, '+V': -1, 'Ak': -1, 'F': -1, 'Fe': -1, 'K': 1,
            'K3': 1, 'Om': -1, 'Tj': -1, 'Tju': -1, 'TR': 1, 'TU': 1,
            'VP': 1, 'X': 1, 'Xs': -1, '/': -1, '': 0, '+L': 1, 'k': 1}

dict_late = {'Ba': -1, '-d': -1, '-l': -1, '-v': 0, '/F': -1, '+D': 0,
             '+I': 0, '+V': -1, 'Ak': -1, 'F': -1, 'Fe': -1, 'K': -1,
             'K3': -1, 'Om': -1, 'Tj': -1, 'Tju': -1, 'TR': -1, 'TU': -1,
             'VP': -1, 'X': -1, 'Xs': -1, '/': -1, '': 0, '+L': 1, 'k': 1}

dict_night = {'Ba': -1, '-d': 0, '-l': -1, '-v': -1, '/F': -1, '+D': -1,
              '+I': -1, '+V': 1, 'Ak': -1, 'F': -1, 'Fe': -1, 'K': -1,
              'K3': -1, 'Om': -1, 'Tj': -1, 'Tju': -1, 'TR': -1, 'TU': -1,
              'VP': -1, 'X': -1, 'Xs': -1, '/': -1, '': 0, '+L': 1, 'k': 1}

# Flatten the array X
X = np.ravel(X)
X = np.core.defchararray.replace(X,' ', '')

# Creates an array size N*M concerning day shift
request_day = [dict_day.get(item, item) for item in X]
request_day = np.resize(request_day, (N, M))

# Creates an array size N*M concerning late shift
request_late = [dict_late.get(item, item) for item in X]
request_late = np.resize(request_late, (N, M))

# Creates an array size N*M concerning night shift
request_night = [dict_night.get(item, item) for item in X]
request_night = np.resize(request_night, (N, M))

for row in range(N):
    if nightshift[row] == 0:
        for col in range(M):
            
            # some employees can wish for night shifts, but don't have to
            if request_night[row, col] < 1:
                request_night[row, col] = -1
        
        # senior physisians
        if competence[row] == 1:
            request_late[row, :] = -1

request_night[request_night < 0] = -fit_var['PEN_HARD']

# Employees that wish for day shift or late shift can't have night shift the day before
for row in range(N):
    for col in range(1, M):
        if request_day[row, col] > 0 or request_late[row, col] > 0:
            request_night[row, col-1] = -fit_var['PEN_HARD']

# Resize the array X
X = np.resize(X, (N, M))

# Assemble the 3 arrays in to a N*M*3 dimensional array
requests = np.dstack((request_day, request_late, request_night))


# *****************************************************************************
# define functions ************************************************************
# *****************************************************************************


def randPopulation(pop_size=model_var['POP_SIZE']):
    """
    Creates a population with some randomness. The starting point for every 
    member of the population is the given requests. The rest is filled out by
    random. 
    """

    if pop_size == model_var['POP_SIZE']:
        print("Creating initial population...")

    # Initialize the population
    population = []

    for _ in range(pop_size):

        # The starting point is requests from the employees
        individual = np.copy(requests)

        # Distribute night shifts
        for col in range(M):
            while (individual[:, col, 2] > 0).sum() != fit_var['N_NIGHT']:

                # Delete night shifts
                if (individual[:, col, 2] > 0).sum() > fit_var['N_NIGHT']:
                    random_row = random.randint(0, N-1)
                    individual[random_row, col, 2] = 0

                # Add night shifts
                if (individual[:, col, 2] > 0).sum() < fit_var['N_NIGHT']:
                    random_row = random.randint(0, N-1)
                    if (individual[random_row, col, 2] == 0 and 
                        nightshift[random_row] == 1):
                        individual[random_row, col, 2] = 1

            # Block work after night shift
            if col < M-1:
                for row in range(N):
                    if individual[row, col, 2] > 0:
                        individual[row, col, :2] = -1
                        individual[row, col+1, :] = -1
            if col == M-1:
                for row in range(N):
                    if individual[row, col, 2] > 0:
                        individual[row, col, :2] = -1

        # Distribute late shifts
        for col in range(M):
            while (individual[:, col, 1] > 0).sum() != fit_var['N_LATE'] * (1-weekend[col]):

                # Delete late shifts
                if (individual[:, col, 1] > 0).sum() > fit_var['N_LATE'] * (1-weekend[col]):
                    random_row = random.randint(0, N-1)
                    individual[random_row, col, 1] = 0

                # Add late shifts
                if (individual[:, col, 1] > 0).sum() < fit_var['N_LATE'] * (1-weekend[col]):
                    random_row = random.randint(0, N-1)
                    if individual[random_row, col, 1] == 0:
                        individual[random_row, col, 1] = 1

        # Distribute weekend shifts
        for col in range(M):
            if weekend[col] == 1:
                while (individual[:, col, 0] > 0).sum() != fit_var['N_WEEKEND']:

                    # Delete weekend shifts
                    if (individual[:, col, 0] > 0).sum() > fit_var['N_WEEKEND']:
                        random_row = random.randint(0, N-1)
                        individual[random_row, col, 0] = 0

                    # Add weekend shifts
                    if (individual[:, col, 0] > 0).sum() < fit_var['N_WEEKEND']:
                        random_row = random.randint(0, N-1)
                        if individual[random_row, col, 0] == 0:
                            individual[random_row, col, 0] = 1
            else:
                continue


        # Distribute day shifts
        for row in range(N):
            HOURS_NIGHT = (individual[row, :, 2] > 0).sum() * fit_var['HOURS_NIGHT']
            HOURS_LATE = (individual[row, :, 1] > 0).sum() * fit_var['HOURS_LATE']
            remaining_hours = hours_to_deliver[row] - (individual[row, :, 0] > 0).sum() * fit_var['HOURS_DAY'] - HOURS_NIGHT - HOURS_LATE
            n_day_shift = max(0, int(remaining_hours/fit_var['HOURS_DAY'])) + fit_var['EXTRA_WEEKDAYS']

            counter = 0

            while (individual[row, :, 0] > 0).sum() != n_day_shift and counter < 1000:

                # Delete day shifts
                if (individual[row, :, 0] > 0).sum() > n_day_shift:
                    random_col = random.randint(0, M-1)

                    # Don't delete weekend shift
                    if weekend[random_col] == 0:
                        individual[row, random_col, 0] = 0

                # Add day shifts
                if (individual[row, :, 0] > 0).sum() < n_day_shift:
                    random_col = random.randint(0, M-1)
                    if individual[row, random_col, 0] == 0 and individual[row, random_col, 1] < 1 and weekend[random_col] == 0:
                        individual[row, random_col, 0] = 1

                counter += 1

        individual[individual > 1] = 1
        individual[individual < 0] = 0

        population.append(individual)

    return population


def fitness(individual, array=False):
    """Calculates the fitness of *individual*.
    Input: N*M*3 dimensional binary numpy array. Layer 0 is day shift,
    layer 1 is late shift and layer 2 is night shift.
    Output: If array=False the output is a number, otherwise a N*M dimensional
    numpy array"""

    # Initialize the penalty array
    pen = np.array(np.zeros((N, M), dtype=float))

    # sum of people in day-shift has to be in range 'DAY_MIN' - 'DAY_MAX'
    # or equal to 'weekend' in weekends
    sumPerDay = individual.sum(axis=0)

    # penalty for not having 1 chief physician in day shift
    # and more than 1 junior physician in weekends
    chiefPhys = np.array(competence)
    chiefPhys[chiefPhys != 1] = 0
    chiefPhysPerDay = np.multiply(individual[:, :, 0], chiefPhys[:, np.newaxis]).sum(axis=0)
    juniorPhys = np.array(competence)
    juniorPhys[juniorPhys != 3] = 0
    juniorPhys[juniorPhys == 3] = 1
    juniorPhysPerDay = np.multiply(individual[:, :, 0], juniorPhys[:, np.newaxis]).sum(axis=0)

    # rules that apply to columns
    for j in range(M):
        if weekend[j] == 1:
            # number dayshift
            if sumPerDay[j, 0] != fit_var['N_WEEKEND']:
                pen[:, j] += fit_var['PEN_HARD']/N
            # number nightshift
            if sumPerDay[j, 2] != fit_var['N_NIGHT']:
                pen[:, j] += fit_var['PEN_HARD']/N
            # number of chief physicians present
            if chiefPhysPerDay[j] < fit_var['CHIEF_PHYS_MIN']:
                pen[:, j] += fit_var['PEN_HARD']/N
            # number of junior physicians present
            if juniorPhysPerDay[j] > fit_var['JUNIOR_PHYS_MAX']:
                pen[:, j] += fit_var['PEN_HARD']/N
        else:
            # number dayshift
            if sumPerDay[j, 0] > fit_var['DAY_MAX']:
                pen[:, j] += fit_var['PEN_MEDIUM']/N
            if sumPerDay[j, 0] < fit_var['DAY_MIN']:
                pen[:, j] += fit_var['PEN_HARD']/N
            # number nightshift
            if sumPerDay[j, 2] != fit_var['N_NIGHT']:
                pen[:, j] += fit_var['PEN_HARD']/N
            # number late shift
            if sumPerDay[j, 1] != fit_var['N_LATE']:
                pen[:, j] += fit_var['PEN_MEDIUM']/N
            # number of chief physicians present
            if chiefPhysPerDay[j] < fit_var['CHIEF_PHYS_MIN']:
                pen[:, j] += fit_var['PEN_HARD']/N

    # penalty for failing to comply with requests
    pen += requestsFulfilled(individual) * fit_var['PEN_MEDIUM']

    # calculate how many day/late/night shifts each employee has
    sumPerPerson = individual.sum(axis=1)
    
    # calculate how many night shifts each employee should cover in average
    nights_per_person = (M * fit_var['N_NIGHT'] - 3) / nightshift.count(1)
    nights_floor = math.floor(nights_per_person)
    nights_ceil = math.ceil(nights_per_person)

    # rules that apply to rows
    for row in range(N):
        # sum hours worked and number of night shifts
        sumHours = sumPerPerson[row, 0] * fit_var['HOURS_DAY'] + sumPerPerson[row, 1] * fit_var['HOURS_LATE'] + sumPerPerson[row, 2] * fit_var['HOURS_NIGHT']
        n_nights = sumPerPerson[row, 2]

        # penalty for working too much/little
        if abs(sumHours - hours_to_deliver[row]) > fit_var['TOLERANCE_HOUR']:
            pen[row, :] += (abs(sumHours - hours_to_deliver[row]) - fit_var['TOLERANCE_HOUR'])  * fit_var['PEN_SOFT']/M
       
        # penalty for having too many/few night shifts
        if (n_nights < nights_floor or n_nights > nights_ceil) and nightshift[row] != 0:
            pen[row, :] += abs(n_nights - 4) * fit_var['PEN_HARD']/M


    # penalty for having 2 or more night shifts in a row: [1,1]
    seq = np.array([1, 1])
    pen += searchSequence(individual, seq, 2) * fit_var['PEN_HARD']

    # penalty for having 2 or more late shifts in a row: [1,1] 
    pen += searchSequence(individual, seq, 1) * fit_var['PEN_HARD']

    # penalty for working too many days in a row: [1,1,1,1,1,1,1,1]
    # IT NEEDS TO ALSO INCLUDE NIGHT SHIFTS!
    seq = np.array([1, 1, 1, 1, 1, 1, 1, 1])
    pen += searchSequence(individual, seq, 0) * fit_var['PEN_HARD']

    # penalty for working after night shift
    night = individual[:, :, 2]
    day = individual[:, :, 0] + individual[:, :, 1]

    # add a column zeroes in front of 'night' and ind the back of 'day'
    night = np.concatenate((np.zeros((N, 1)), night), 1)
    day = np.concatenate((day, np.zeros((N, 1))), 1)

    violation = np.multiply(day, night)

    # delete the first column
    violation = np.delete(violation, -1, axis=1)

    pen += violation * fit_var['PEN_HARD']

    if array:
        return pen
    else:
        return pen.sum()


def requestsFulfilled(individual):
    """Helper function for the fitness function. This function checks if
    requests are fulfilled. 2 conditions out the 6 possible gives a penalty
    (see below).

    individual   requests   violation
    [0 0 0]      [0 1 -1]   [0 1 0]
    [1 1 1]      [0 1 -1]   [0 0 1]
    """

    temp = np.array(np.zeros((N, M, 3), dtype = float))

    # individual = 1, request = -1
    violation = (np.multiply(individual, requests))*-1

    # individual = 0, request = 1
    temp[individual == 1] = 0
    temp[individual == 0] = 1
    temp = np.multiply(temp, requests)
    violation = np.add(temp, violation)

    # remove the negative numbers
    violation[violation < 0] = 0

    # choose the highest violation
    violation = violation.max(axis=2)

    return violation


def searchSequence(individual, seq, layer):
    """
    Helper function for the fitness function. Finds a sequence in *individual*
    INSPIRED BY:
    http://stackoverflow.com/questions/36522220/searching-a-sequence-in-a-numpy-array
    """

    individual = individual[:, :, layer]
 
    # Store sizes of input array and sequence
    output = np.array(np.zeros((N, M)))

    seq_length = len(seq)

    for n in range(N):

        row = individual[n, :]
        # Range of sequence
        seq_range = np.arange(seq_length)

        # Create 2D array of sliding indices across entire length of input array.
        # Match up with the input sequence & get the matching starting indices.
        arr = (row[np.arange(M-seq_length+1)[:, None] + seq_range] == seq).all(1)

        indices = np.where(np.convolve(arr, np.ones((seq_length), dtype=int)) > 0)[0]

        for m in indices:
            output[n, m] = 1

    return output


def permuteTwins(population):
    """
    This function looks for twins in population.
    If it finds a twin, it permutes one of the twins at the column with the
    highest penalty.
    """
    unique = []

    for n, ind1 in enumerate(population):
        if any(np.array_equal(ind1, ind2) for ind2 in unique):
            fit = fitness(ind1,array=True)
            col = np.argmax(np.max(fit, axis=1))
            np.random.shuffle(population[n][:, col])
            continue
        unique.append(ind1)

    return population
    
    
def removeTwins(population):
    """
    This function looks for twins in population. Twins are removed and
    new random individuals are inserted. This procedure is performed to
    maintain the populations diversity.
    """
    global twins_removed

    # Create a list of unique individuals from population
    unique = []
    for ind1 in population:
        if any(np.array_equal(ind1, ind2) for ind2 in unique):
            continue
        unique.append(ind1)

    # Calculate how many twins has been removed
    twins_removed = len(population) - len(unique)
    
    # Replace the removed twins with new random individuals
    population = randPopulation(twins_removed)
    population.extend(unique)

    return population
    

def weightedPopulation(population):
    """
    Creates a weighted population by making a list of lists [individual, weight, fitness]
    """
    weighted_population = []
    
    for individual in population:
        fit = fitness(individual)
        weighted_population.append([individual, 1/(fit+1), fit])
    
    weighted_population.sort(key=lambda x: x[2])

    return weighted_population


def selection(weighted_population, k=model_var['N_TOURNAMENT']):
    """
    Select an individual from weighted_population, where weighted_population is a list of tuples in
    the form (item, weight). If 'TOURNAMENT' is True the individual is selected
    by the tournament method, otherwise by the roulette method.
    """

    # Tournament method
    if model_var['TOURNAMENT']:
        selected = None
        for _ in range(k):
            individual = weighted_population[random.randrange(0, len(weighted_population))]
            if selected is None or individual[1] > selected[1]:
                selected = individual
        return selected[0]

    # Roulette method
    else:
        weights = [ind[1] for ind in weighted_population]
        selected = random.choices(weighted_population, weights=weights, k=1)
        return selected[0][0]
    

def crossover(parent1, parent2):
    """
    1-point crossover. Slices parent1 and parent2 into two parts at a random
    crossover index (horizontally). Both keep their initial start and get the
    end from the other.
    """
    
    # Choose random column position between 1 and M-1
    pos = random.randint(1, M-1)
    
    # Split the parents at the random column
    temp1 = np.split(parent1, [pos], axis=1)
    temp2 = np.split(parent2, [pos], axis=1)
    
    # Create the children by collecting the parts from the parents 
    child1 = np.concatenate((temp1[0], temp2[1]), axis=1)
    child2 = np.concatenate((temp2[0], temp1[1]), axis=1)
    
    return child1, child2


def crossoverTwoPoint(parent1, parent2):
    """
    2-point crossover. The two individuals are modified in place and both keep
    their original length.
    """

    # Choose 2 random column positions between 1 and M-1
    pos1 = random.randint(1, M-1)
    pos2 = random.randint(1, M-1)

    # Swap the two crossover points
    if pos2 < pos1:
        pos1, pos2 = pos2, pos1

    # Split the parents at the random columns
    temp1 = np.split(parent1, [pos1,pos2], axis=1)
    temp2 = np.split(parent2, [pos1,pos2], axis=1)
    
    # Create the children by collecting the parts from the parents 
    child1 = np.concatenate((temp1[0], temp2[1], temp1[2]), axis=1)
    child2 = np.concatenate((temp2[0], temp1[1], temp2[2]), axis=1)

    return child1, child2


def mutate(individual,mutate=model_var['MUTATE'],permute=model_var['PERMUTE']):
    """Function that makes a 'random' mutation and/or permutation in an array.
    The probability of permutation is 1 / permutation_rate.
    The probability of mutation is 1 / mutation_rate.
    Mutations are only made if mutate=True.
    Permutations are only made if permutation=True"""

    if permute:
        for col in range(M):
            if int(random.random()*model_var['PERMUTATION_RATE']) == 1:
                np.random.shuffle(individual[:, col])

    if mutate:
        mutation = False
        if int(random.random()*model_var['MUTATION_RATE']) == 1:
            while mutation is False:
                row = random.randrange(0, N, 1)
                col = random.randrange(0, M, 1)
                if individual[row, col, 2] != 1 and individual[row, col, 1] != 1:
                    individual[row, col, 0] = random.randrange(0, 1, 1)
                    mutation = True

    return individual


def elitism(weighted_population, n_elitism=model_var['N_ELITISM']):
    """
    Selects the best N_ELITISM individuals from the weighted population. The
    weighted population is already sorted with the best individuals first.
    """
    
    ranked_individuals = [ind[0] for ind in weighted_population]
    
    return ranked_individuals[:n_elitism]


def localSearch(weighted_population):
    """
    This function try to optimize the best candidate roster when the genetic 
    algorithm is stuck. The function either search through all of the roster, or
    only try to repair the worst part. 
    """
    
    new_weighted_population = []
    
    # Try to improve all of the roster - very slow
    if model_var['SEARCH_ALL']:
        
        # Loop through the weighted population
        for weighted_individual in weighted_population:
            
            # Extract individual and fitness score from the list
            individual = weighted_individual[0]
            fit_before = weighted_individual[2]
            
            # Make a hard copy of the individual
            new_individual = np.copy(individual)
    
            # Loop through the entire roster
            for row in range(N):
                for col in range(M):
                    
                    # Make a change only if no night/late shift is assigned
                    if individual[row, col, 1] != 1 and individual[row, col, 2] != 1:
                        new_individual[row, col, 0] = 1 - individual[row, col, 0]
                        fit_after = fitness(new_individual)
                        
                        # Keep the change if the fitness score is better than before
                        if  fit_after <= fit_before:
                            individual = np.copy(new_individual)
                            fit_before = fit_after
                        else:
                            new_individual = np.copy(individual)
            
            new_weighted_population.append([individual, 1/(fit_before+1), fit_before])
        
    # Only try to improve the worst part of the roster - much faster
    else:
        
        # Loop through the weighted population
        for weighted_individual in weighted_population:
            
            # Extract individual and fitness score from the list
            individual = weighted_individual[0]
            fit_before = weighted_individual[2]
            
            # Make a hard copy of the individual
            new_individual = np.copy(individual)
    
            # Calculate the fitness array
            fit_array = fitness(individual, array=True)
            
            # Find the row,col with the maximum penalty
            row = np.argmax(np.max(fit_array, axis=1))
            col = np.argmax(np.max(fit_array, axis=0))
            
            # Make a change only if no night/late shift is assigned
            if individual[row, col, 1] != 1 and individual[row, col, 2] !=1:
                new_individual[row, col, 0] = 1 - individual[row, col, 0]
    
            fit_after = fitness(new_individual)
    
            # Keep the change if the fitness score is better than before
            if  fit_after < fit_before:
                new_weighted_population.append([new_individual, 1/(fit_after+1), fit_after])
            else:
                new_weighted_population.append([individual, 1/(fit_before+1), fit_before])
    
    # Sort the new weighted population by fitness score
    new_weighted_population.sort(key=lambda x: x[2])
    return new_weighted_population
            

def hoursDelivered(individual):
    """Function that calculates how many hours each employee delivers"""
    
    delivered = np.array(np.zeros((N), dtype=int))
    
    for row in range(N): 
        delivered[row] += (best_individual[row, :, 0] == 1).sum() * fit_var['HOURS_DAY']
        delivered[row] += (best_individual[row, :, 1] == 1).sum() * fit_var['HOURS_LATE']
        delivered[row] += (best_individual[row, :, 2] == 1).sum() * fit_var['HOURS_NIGHT']

    return delivered


def binaryToPlan(individual, fit_array):
    """
    Converts a N*M*3 binary array to a N*M array
    """
    
    # Allocate space for the new arrays
    best_plan = np.array(np.empty((N, M), dtype = '|U5'))
    details = np.array(np.empty((N, M), dtype = '|U5'))
    
    # Codes corresponding to day shift
    day_work_set = {'Ak', 'K', 'K3', 'TR', 'TU', 'VP', 'X', 'Xs'}
    
    # Codes corresponding to day off
    day_off_set = {'Ba', 'Fe', 'Om', 'Tj', 'Tju'}
    
    # Loop through the entire roster
    for row in range(N):
        for col in range(M):
            
            # If day shift is assigned
            if individual[row,col,0] == 1:
                if X[row,col] in day_work_set:
                    best_plan[row,col] += X[row,col]
                else:
                    best_plan[row,col] += 'd'
            
            # If late shift is assigned
            if individual[row,col,1] == 1:
                best_plan[row,col] += 'l'
              
            # If night shift is assigned
            if individual[row,col,2] == 1:
                best_plan[row,col] += 'n'  
            
            # Set to 'day off' if still not assigned to any shift
            if best_plan[row, col] == '':
                if X[row,col] in day_off_set:
                    best_plan[row,col] += X[row,col]
                else:
                    best_plan[row, col] = 'f'
            
            # The 'details' array contains information about requests
            if model_var['DEBUG'] and X[row, col] != "":
                details[row, col] += "/" + X[row, col]
            else:
                if fit_array[row,col] >= fit_var['PEN_MEDIUM']:
                    details[row, col] += "/" + X[row, col]
    
    return best_plan, details


def showBestPlan(individual):
    """
    Input: Index of the best individual in the population
    Output: 1) Heatmap that shows the best roster.
            2) Plot generation vs. fitness
    """
    
    fit_array = fitness(individual, array=True)
    plan, details = binaryToPlan(individual, fit_array)
    
    # calculate the hour balance for each employee
    hour_difference = hoursDelivered(best_individual) - hours_to_deliver
    hour_difference = [int(i) for i in hour_difference]
    
    # count how many employees contribute in the clinic per day
    n_day = [(plan[:, col] == 'd').sum() for col in range(M)]
    n_day.append("")

    for row in range(N):
        for col in range(M):
            plan[row, col] += details[row, col]
            
    # add column to plan
    plan = np.c_[plan, hour_difference] 
    
    # add row to plan
    plan = np.vstack([plan, n_day])
    
    column_labels = dates
    row_labels = names

    fig, ax = plt.subplots(figsize=(20,10))

    #heatmap = ax.pcolor(fit, alpha = 0.7, cmap=plt.cm.Reds)
    heatmap = ax.pcolor(fit_array, alpha = 0.7, cmap=plt.cm.Blues)

    # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(N)+0.5, minor=False)
    ax.set_xticks(np.arange(M+1)+0.5, minor=False)

    # set labels
    ax.set_xticklabels(column_labels, minor=False, rotation=80)
    ax.set_yticklabels(row_labels, minor=False)

    # inverse the y-axis and put the column labels on the top
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    
    for row in range(N+1):
        for col in range(M+1):
            plt.text(col + 0.5, row + 0.5, '%s' % plan[row, col],
                     horizontalalignment='center',  
                     verticalalignment='center',
                     )

    ax.axis('tight')
    plt.colorbar(heatmap)
    
    # set the x-tick_label to red if the date is in a weekend
    weekend.append(0) 
    [handle.set_color('red') for i, handle in enumerate(plt.gca().get_xticklabels()) if weekend[i] == 1]
    
    # show plot
    plt.show(block=True)
    
    # plot how the score develops over GENERATIONS
    plt.figure()
    plt.semilogy(generation_list, score_list)
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.show(block=True)    
    
    
def geneticAlgorithm(start_population=None):
    """
    Main function that runs the genetic algorithm. If *start_population* is None
    (default) the function will generate a population, but it is also possible
    to give a start population as an argument.
    """
    
    # Initialize lists and variables
    global score_list
    global generation_list
    score_list = []
    generation_list = []
    last_score = math.inf
    no_improvement = 0

    if start_population is None:
        # Generate the initial population
        population = randPopulation()
    else:
        # Use population given as argument
        population = start_population

    # Run the genetic algorithm for *GENERATIONS* generations
    for generation in range(model_var['GENERATIONS']):
        
        # permute twins
        if model_var['PERMUTE_TWINS']:
            population = permuteTwins(population)

        # remove twins and replace with new random individuals
        if model_var['REMOVE_TWINS']:
            population = removeTwins(population)
        
        # Create a weighted population
        weighted_population = weightedPopulation(population)
        
        # Extract the best fitness score
        best_score = int(weighted_population[0][2])
        score_list.append(best_score)
        generation_list.append(generation)
        
        # Stop the algorithm if the best score has been reached
        if best_score == 10:
                print("The best possible score has been reached")
                break
            
        # Count how many genrations have passed since the last improvement
        if best_score < last_score:
            last_score = best_score
            no_improvement = 0
        else:
            no_improvement += 1
        
        # Stop the algorithm if it is stuck
        if no_improvement >= model_var['STOP_CRITERION']:
            print("There algorithm has stopped due to lack of improvement!")
            break
        
        # Try local search if LOCAL_SEARCH_CRITERION generations have passed since the last improvement 
        if model_var['LOCAL_SEARCH'] and no_improvement == model_var['LOCAL_SEARCH_CRITERION']:
            print("At generation", generation, ": No improvement for",no_improvement,"generations!")
            print("Trying local search (make some coffee)...")

            weighted_population = localSearch(weighted_population)
            print("Local search finished. Continuing the genetic algorithm...")
                
        # print the best score for every PRINT_GENERATION generations
        if (generation % model_var['PRINT_GENERATION']) == 0:
        
            print("Generation: ", generation, "/", model_var['GENERATIONS'],". Best score: ", best_score)

            if model_var['DEBUG']:
                
                # Calculate and print the "diversity" if in debug-mode
                total_diversity = sum([np.sum(x != y) for x in population for y in population if x is not y])
                print("Diversity: ", total_diversity/(model_var['POP_SIZE']**2-model_var['POP_SIZE']))
                if model_var['REMOVE_TWINS']:
                    print("Twins removed: ",twins_removed)

        # Delete the old population so a new one can be made
        population = []
        
        # Perform elitism by transfering the N_ELITISM fittest individuals to the next generation
        if model_var['ELITISM']:
            population = elitism(weighted_population)
        
        # Add individuals to the new population until POP_SIZE has been reached
        while len(population) < model_var['POP_SIZE']:
            
            # Selection
            father = selection(weighted_population)
            mother = selection(weighted_population)

            # Crossover
            if model_var['TWO_POINT_CROSSOVER']:
                child1, child2 = crossoverTwoPoint(father, mother)
            else:
                child1, child2 = crossover(father, mother)

            # Mutate
            child1 = mutate(child1)
            child2 = mutate(child2)

            # Accept
            population.append(child1)
            population.append(child2)
            
    return population

# *****************************************************************************
# MAIN PROGRAM ****************************************************************
# *****************************************************************************

start = datetime.datetime.now()
"""
if model_var['DEBUG']:
    np.random.seed(81)  # Perfect roster for POP_SIZE 100
    random.seed(81)     
"""
# Run the genetic algorithm
population = geneticAlgorithm()

# Extract the best score and the best individual from the population
weighted_population = weightedPopulation(population)
best_individual = weighted_population[0][0]
best_score = int(weighted_population[0][2])

# Display the best individual after all generations have been iterated over.
print("The best roster has the score:", best_score)
if best_score > fit_var['PEN_HARD']:
    print("The best roster is not feasible.")
    print("Number of hard constraints violated: ",int(best_score/fit_var['PEN_HARD']))
showBestPlan(best_individual)

# Save the roster with the name "roster-YYYY-MM-DD-HHMM-score-##.npy"
date_string = time.strftime("%Y-%m-%d-%H%M")
np.save('roster-' + date_string + '-score-' + str(best_score) + '.npy', best_individual)

print('Time spend:', datetime.datetime.now()-start)

# *****************************************************************************
# JUST FOR DEVELOPMENT ********************************************************
# *****************************************************************************

"""
n = 0
best_score = 100

while best_score > 10:
    print("SEED:", n)
    np.random.seed(n)
    random.seed(n)
    population = geneticAlgorithm()
    weighted_population = weightedPopulation(population)
    best_individual = weighted_population[0][0]
    best_score = int(weighted_population[0][2])
    n += 1
"""

# *****************************************************************************
# FUNCTIONS NOT IN USE AT THE MOMENT ******************************************
# *****************************************************************************

"""
def searchSequence2(individual, length, layer):
    
    #Helper function for the fitness function. Finds a sequence in *individual*
    #It is VERY SLOW, but it works!
    
    np.set_printoptions(threshold=3000)

    # Choose the specified layer
    individual = individual[:, :, layer]

    # Initialize the output array
    output = np.array(np.zeros((N, M)))

    # Loop through the rows
    for row in range(N):

        # Convert the row to a string of '0' and '1's
        row_str = np.array_str(individual[row, :])
        row_str = row_str.replace('[', '').replace(']', '').replace(' ', '')

        col = 0
        n_ones = 0
        for i in range(len(row_str)):
            if row_str[i] == '1':
                n_ones += 1
                if n_ones >= length:
                    output[row,col] = 1
            else:
                n_ones = 0
            col += 1

    return output
"""