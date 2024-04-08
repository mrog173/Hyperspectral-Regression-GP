import random
from deap import gp, tools
import sys
from functools import partial
import numpy as np
from inspect import isclass
from timeit import default_timer as timer
from utils.primitives import SpectralFeature, FeatureVector, ReflectanceSpectra, GreyscaleImg
from utils.generate_outputs import *

__type__ = object

def mutUniform(individual, expr, pset, min_depth, max_depth):
    """Randomly select a point in the tree *individual*, then replace the
    subtree at that point as a root by the expression generated using method
    :func:`expr`.

    individual: The tree to be mutated.
    expr: A function object that can generate an expression when called.
    returns: A tuple of one tree.
    """
    #Select a random point that isn't a terminal
    index = -1
    while individual[index].arity == 0:
        index = random.randrange(len(individual))

    slice_ = individual.searchSubtree(index)
    full_depth = individual.height
    subtree_depth = gp.PrimitiveTree(individual[slice_]).height
    point_depth = full_depth-subtree_depth

    type_ = individual[index].ret
    individual[slice_] = expr(pset=pset, type_=type_, min_=max(min_depth-point_depth,0), max_=max_depth-point_depth)
    return individual,

def genGrow(pset, min_, max_, type_=None):
    """Generate an expression where each leaf might have a different depth between *min* 
    and *max*.

    pset : Primitive set from which primitives are selected.
    min_ : Minimum height of the produced trees.
    max_ : Maximum Height of the produced trees.
    type_ : The type that should return the tree when called, when `None` (default) no 
            return type is enforced.
    returns : A grown tree with leaves at possibly different depths.
    """
    if type_ is None:
        type_ = pset.ret
    expr = []

    height = random.randint(min_, max_)

    #Define sets
    transition_set = ["mean_interval_selection", "median_interval_selection", 'img_interval_median', 'img_interval_selection', 'img_interval_selection_gaussian', "convert_to_feature", "mean_feature", "mean_spectra_extraction"]
    full_set = ['img_interval_selection_median', 'img_interval_selection', 'img_interval_selection_gaussian', 'mean_spectra_extraction']
    stages = {SpectralFeature:2, FeatureVector:3, GreyscaleImg:1, ReflectanceSpectra:1}

    stack = [(0, type_)]
    while len(stack) != 0:
        depth, type_ = stack.pop()
        try:
            # Select any primitives as long as the depth is within
            if depth < min_-stages.get(type_,0): # Don't select a final primitive if the branch isn't large enough
                possible_functions = [x for x in pset.primitives[type_] if x.name not in full_set] + pset.terminals[type_]
                term = random.choice(possible_functions)
            # If current depth is less than height - 
            elif depth < height - stages.get(type_,0): # Continue randomly growing
                term = random.choice(pset.terminals[type_] + pset.primitives[type_])
            else: # Make transition towards the leaves
                possible_functions = [x for x in pset.primitives[type_] if x.name in transition_set] + pset.terminals[type_]
                term = random.choice(possible_functions)
        except IndexError:
            _, _, traceback = sys.exc_info()
            raise IndexError("The gp.generate function tried to add a terminal of type '%s', but there is none available." % (type_,)).with_traceback(traceback)
        
        # Generate a value for the terminal
        if isclass(term):
            term = term()

        # Add primitives to stack
        elif term in pset.primitives[type_]:
            for arg in reversed(term.args):
                stack.append((depth + 1, arg))
        expr.append(term)
    return expr

def genFull(pset, min_, max_, type_=None):
    """Generate an expression where each leaf has a the same depth between *min* and 
    *max*.

    pset : Primitive set from which primitives are selected.
    min_ : Minimum height of the produced trees.
    max_ : Maximum Height of the produced trees.
    type_ : The type that should return the tree when called, when `None` (default) no 
            return type is enforced.
    returns : A full tree with all leaves at the same depth.
    """
    if type_ is None:
        type_ = pset.ret
    expr = []
    
    height = random.randint(min_, max_)
    # Define sets
    transition_set = ["mean_interval_selection", "median_interval_selection", 'img_interval_median', 'img_interval_selection', 'img_interval_selection_gaussian', "convert_to_feature", "mean_feature", "mean_spectra_extraction"]
    full_set = ['img_interval_selection_median', 'img_interval_selection', 'img_interval_selection_gaussian', 'mean_spectra_extraction']
    stages = {SpectralFeature:2, FeatureVector:3, GreyscaleImg:1, ReflectanceSpectra:1}

    stack = [(0, type_)]
    while len(stack) != 0:
        depth, type_ = stack.pop()
        try:
            # Only select the final primitives when close enough to the target depth
            if depth == height - stages.get(type_,0): # Make transition towards the leaves
                possible_functions = [x for x in pset.primitives[type_] if x.name in transition_set] + pset.terminals[type_]
                term = random.choice(possible_functions)
            else: # Don't select a final primitive if the branch isn't large enough
                possible_functions = [x for x in pset.primitives[type_] if x.name not in full_set] + pset.terminals[type_]
                term = random.choice(possible_functions)

        except IndexError:
            _, _, traceback = sys.exc_info()
            raise IndexError("The gp.generate function tried to add a terminal of type '%s', but there is none available." % (type_,)).with_traceback(traceback)
        
        # Generate a value for the terminal
        if isclass(term):
            term = term()

        # Add primitives to stack
        elif term in pset.primitives[type_]:
            for arg in reversed(term.args):
                stack.append((depth + 1, arg))
        expr.append(term)
    return expr

def genHalfAndHalf(pset, min_, max_, type_=None):
    """Generate an expression with a PrimitiveSet *pset*. Half the time, the expression 
    is generated with the grow method of generation, the other half, the expression is 
    generated with the

    pset : Primitive set from which primitives are selected.
    min_ : Minimum height of the produced trees.
    max_ : Maximum Height of the produced trees.
    type_ : The type that should return the tree when called, when :obj:`None` (default) 
            no return type is enforced.
    returns : Either, a full or a grown tree.
    """
    method = random.choice((genGrow, genFull))
    return method(pset, min_, max_, type_)

def varOr(population, toolbox, cxpb, mutpb, mutephpb):
    """Part of an evolutionary algorithm applying only the variation part (crossover 
    **or** mutation). The modified individuals have their fitness invalidated. The 
    individuals are cloned so returned population is independent of the input 
    population.

    population : A list of individuals to vary.
    toolbox : A :class:`~deap.base.Toolbox` that contains the evolution operators.
    cxpb : The probability of mating two individuals.
    mutpb : The probability of mutating an individual.
    elitpb : The probability of mutating an individual.
    returns : A list of varied individuals that are independent of their parents.
    """
    offspring = [toolbox.clone(ind) for ind in population]
    random.shuffle(offspring)

    # Apply crossover
    i=1
    while i<len(offspring):
        val = random.random()
        if val < cxpb:
            offspring[i-1], offspring[i] = toolbox.mate(offspring[i-1], offspring[i])
            del offspring[i-1].fitness.values, offspring[i].fitness.values
            i += 2
        elif val < cxpb+mutpb:
            val2 = random.random()
            if val2 < mutephpb:
                offspring[i], = toolbox.mutate_eph(offspring[i])
            else:
                offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values
            i += 1

    return offspring

def eaSimple(population, toolbox, config, REPORT_PATH, stats=None, halloffame=None, verbose=__debug__, iteration=0):
    """Based on the implementation provided in DEAP.
    This algorithm reproduce the simplest evolutionary algorithm as presented in 
    chapter 7 of Back, Fogel and Michalewicz, "Evolutionary Computation 1 : Basic 
    Algorithms and Operators", 2000.

    population : A list of individuals.
    toolbox : A :class:`~deap.base.Toolbox` that contains the evolution operators.
    config : A dictionary containing all the 
    REPORT_PATH : The path of the folder to output all report values to 
    stats : A `~deap.tools.Statistics` object that is updated inplace.
    halloffame : A `~deap.tools.HallOfFame` object that will contain the best individuals.
    verbose : Whether or not to log the statistics.
    returns : The final population and a `~deap.tools.Logbook` with the statistics of the evolution
    """
    logbook = tools.Logbook()
    start = timer()
    logbook.header = (stats.fields + ['details', 'best'] if stats else [])
    
    fitnesses = toolbox.map(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # Update the hall of fame with the generated individuals
    if halloffame is not None:
        halloffame.update(population)

    #Get fitness score based on validation set
    validation_score = np.zeros(len(halloffame))
    for i, ind in enumerate(halloffame):
        validation_score[i] = partial(toolbox.validation, ind)()['R^2 Testing']

    # Update record dict
    record = {'details':{"gen":0, "nevals":len(population), "time":round(timer()-start, 3)}, 'best':{"size":len(halloffame[0]), "fitness":round(halloffame[0].fitness.values[0],4), "feats":str(halloffame[0]).count("IN0"), "depth":halloffame[0].height, "val_mean":np.round(np.mean(validation_score),4), "val_max":np.round(validation_score[0],4)}}
    record.update(stats.compile(population) if stats else {})
    
    # Output statistics to CSV and terminal
    logbook.record(**record)
    if verbose:
        vals = logbook.stream
        print(vals)
        # Save header and values in CSV
        with open(REPORT_PATH + "/Output.csv", "a") as f:
            f.write(vals.split("\n")[-2].replace(" ", "").replace("\t", ",") + "\n")
            f.write(vals.split("\n")[-1].replace(" ", "").replace("\t", ",") + "\n")

    for gen in range(1, config["generations"] + 1):
        start = timer() # Time how long each iteration takes

        # Duplicate the population 
        population_for_va=[toolbox.clone(ind) for ind in population]

        # Select the next generation individuals by elitism and find the best individual
        elitismNum = max(int(config["elitism_prob"] * len(population)), 1)
        offspringE = toolbox.selectElitism(population_for_va, k = elitismNum)
        halloffame = toolbox.update_hof(toolbox.selectElitism(offspringE, k=1), halloffame)

        with open(REPORT_PATH + "/Best_individuals.txt", "a") as f:
            f.write(str(iteration) + str(gen) + "," + str(halloffame[0]) + "\n")

        # Select the next generation individuals for crossover and mutation
        offspring = toolbox.select(population, len(population) - elitismNum)

        # Vary the pool of individuals and generate the next generation individuals
        offspring = varOr(offspring, toolbox, config["crossover_prob"], config["mutation_prob"], config["mut_eph_prob"])
            
        # Evaluate the individuals with an invalid fitness (new individuals)
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Get fitness score based on validation set
        validation_score = np.zeros(len(halloffame))
        for i, ind in enumerate(halloffame):
            validation_score[i] = partial(toolbox.validation, ind)()['R^2 Testing']

        # Add offspring from elitism into current offspring
        offspring = offspring + offspringE
        population[:] = offspring
            
        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        record['details'] = {"gen":gen, "nevals":len(population), "time":round(timer()-start, 3)}
        record['best'] = {"size":len(halloffame[0]), "fitness":round(halloffame[0].fitness.values[0],4), "feats":str(halloffame[0]).count("IN0"), "depth":halloffame[0].height, "val_mean":round(np.mean(validation_score),4), "val_max":round(validation_score[0],4)}
        logbook.record(**record)
        
        # Output statistics to CSV and terminal
        if verbose:
            vals = logbook.stream
            print(vals)
            # Save values in CSV
            with open(REPORT_PATH + "/" + "Output.csv", "a") as f:
                f.write(vals.split("\n")[-1].replace(" ", "").replace("\t", ",") + "\n")
             
    return population, halloffame