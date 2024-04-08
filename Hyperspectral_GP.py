import operator
import sys
from collections import OrderedDict
import yaml
import scipy.special as sp

import numpy as np
from deap import base, creator, gp, tools
from sklearn import svm
from sklearn import metrics
import random

from gp_functions import eaSimple, genGrow, genHalfAndHalf, mutUniform
from utils.generate_outputs import *
import utils.initialise_primitives as initialise_primitives
from utils.helper_functions import *

import warnings

np.seterr(all='raise')
np.seterr(under='ignore')
sp.seterr(loss='ignore')
warnings.filterwarnings("ignore", message="", category=RuntimeWarning)

cache_table = OrderedDict()
 
def evalRegression(toolbox, config, individual, x_train, y_train):
    """Gives the R^2 for the training set
    individual : a GP tree/individual
    x_train : training image
    y_train : ground truth
    """
    func = toolbox.compile(expr=individual)
    
    # Check if individual is in the cache table
    if config["cache_table"] and str(individual) in cache_table:
        cache_table.move_to_end(str(individual))
        return cache_table[str(individual)],
    
    # Extract all feature vectors
    X = [func(x_train[c]) for c in range(len(y_train))]

    if np.isnan(X).any() or np.all(X == X[0]):
        res = config["default"]
    else:
        X = np.asarray(X)
        # Standardise the input (X) using protected division
        X = np.divide(X-np.mean(X, axis=0, keepdims=True), np.std(X, axis=0, keepdims=True), out=np.zeros_like(X), where=np.std(X, axis=0, keepdims=True)!=0)
        # Fit a SVR model to this data
        regr = svm.SVR(kernel='linear')
        instances_x = config["train_indices"] 
        regr.fit(X[instances_x], y_train[instances_x])
        pred = regr.predict(X[config["evals_indices"]])
        if config["fitness_function"] == "MSE":
            res = metrics.mean_squared_error(y_train[config["evals_indices"]], pred, squared=False)
        elif config["fitness_function"] == "RMSE":
            res = metrics.mean_squared_error(y_train[config["evals_indices"]], pred, squared=True)
        elif config["fitness_function"] == "R^2":
            res = metrics.r2_score(y_train[config["evals_indices"]], pred)
        else:
            raise Exception("Unrecognised fitness function in the config")
    
    # Add fitness to the cache table
    if config["cache_table"]:
        if len(cache_table) > 1024:
            cache_table.popitem(last=False)
        cache_table[str(individual)] = res
    return res,

def table(training_y, test_y):
    #Generate a table of the y values
    print("Split, No. Samples, Mean, Std dev, Range")
    print("Calibration,", len(training_y), ",", round(np.mean(training_y),2), ",", round(np.std(training_y),2), ",", str(np.min(training_y))+"-"+str(np.max(training_y)))
    print("Prediction,", len(test_y), ",", round(np.mean(test_y),2), ",", round(np.std(test_y),2), ",", str(np.min(test_y))+"-"+str(np.max(test_y)))

def main(run, config_file="spectral_testing_config.yml"):
    #Read config file
    with open("configs/" + config_file, 'r') as infile:
        config = yaml.safe_load(infile) 
        config["run"] = run

    # Read data
    x_train, y_train, x_test, y_test = import_data(image_based=config["image_based"], attribute=config["attribute"])
    numbers = list([x for x in range(0, len(x_train))])
    random.shuffle(numbers)
    config["train_indices"] = numbers[:70]
    config["evals_indices"] = numbers[70:]
    table(y_train, y_test)

    # Get path to save outputs
    REPORT_PATH = get_reporting_path()

    #Setup fitness function and default fitness value for failed output
    print("Initialising parameters")
    if config["fitness_function"] == "R^2":
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
        config["default"] = -10
    elif config["fitness_function"] == "MSE" or config["fitness_function"] == "RMSE":
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
        config["default"] = 100
    else:
        raise Exception("Unrecognised fitness function in config")

    if config["image_based"]:
        primitive_set = initialise_primitives.image_based(x_train[0].shape[2])
    else:
        primitive_set = initialise_primitives.spectra_based(x_train[0].shape[0])

    toolbox = base.Toolbox()
    toolbox.register("expr", genHalfAndHalf, pset=primitive_set, min_=config["initial_min_depth"], max_=config["initial_max_depth"])
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=primitive_set)
    toolbox.register("update_hof", update_hof)
    toolbox.register("evaluate", evalRegression, toolbox, config, x_train=x_train, y_train=y_train)
    toolbox.register("validation", evalTesting, toolbox, x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test)

    # Define the selection operator for crossover and mutation
    toolbox.register("select", tools.selTournament, tournsize=config["tournament_size"])

    # Define the selection operator for elitism
    toolbox.register("selectElitism", tools.selBest)

    # Define the crossover operator
    toolbox.register("mate", gp.cxOnePoint)

    # Define the mutation operators
    toolbox.register("expr_mut", genGrow, pset=primitive_set)
    toolbox.register("mutate", mutUniform, expr=toolbox.expr_mut, pset=primitive_set, min_depth=config["initial_min_depth"], max_depth=config["maxDepth"])
    toolbox.register("mutate_eph", gp.mutEphemeral, mode='all')

    # Required to prevent crossover creating invalid trees
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=config["maxDepth"]))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=config["maxDepth"]))

    # Run evolution
    print("Initialising population")
    pop = toolbox.population(config["pop_size"])
    halloffame = tools.HallOfFame(10)
    stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats_size = tools.Statistics(key=len)
    stats_terms = tools.Statistics(key=lambda ind: str(ind).count("IN0"))
    mstats = tools.MultiStatistics(fitness=stats_fit,size=stats_size,features=stats_terms)
    mstats.register("avg", rounded_mean, config["default"])
    mstats.register("std", rounded_std, config["default"])
    mstats.register("min", rounded_min, config["default"])
    mstats.register("max", rounded_max, config["default"])
    
    print("Starting evolution")
    pop, halloffame = eaSimple(pop, toolbox, config, REPORT_PATH, stats=mstats, halloffame=halloffame, verbose=True, iteration=run)
    
    # Save best functions
    with open(REPORT_PATH + "/Halloffame.txt", "a") as f:
        for i in range(len(halloffame)):
            f.write(str(run) + ";" + str(i) + ";" + str(halloffame[i]) + "\n")

    # Evaluate the best function on the training set
    result = evalTesting(toolbox, halloffame[0], x_train, y_train, x_test, y_test)
    print("GP finished:", result)

    generate_report(REPORT_PATH, config)

    if config["save_tree"]:
        # Draw a tree of the best individual (Can be commented out if issues arise)
        save_tree(halloffame[0], REPORT_PATH + "/trees/" + str(run).zfill(2)+"end")

if __name__ == "__main__":
    if len(sys.argv) > 2:
        main(sys.argv[1], config_file=sys.argv[2])
    elif len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main(0)
