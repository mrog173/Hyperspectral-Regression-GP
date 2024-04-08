from deap import gp
import yaml

import os
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False

def save_tree(individual, idx):
    """Draw and save the GP function tree as a PDF. Formats and colours the nodes of the 
    tree to make it easier to read.
    """
    import pygraphviz as pgv #Can be commented out, along with the save_tree() function calls if problems arise
    nodes, edges, labels = gp.graph(individual)
    for k in labels.keys():
        if labels[k] == 'convert_to_feature':
            labels[k] = "="
        elif labels[k] == 'median_interval_selection':
            labels[k] = "Median"
        elif labels[k] == 'mean_interval_selection':
            labels[k] = "Mean"
        elif labels[k] == 'addition':
            labels[k] = "+"
        elif labels[k] == 'subtraction':
            labels[k] = "-"
        elif labels[k] == 'multiplication':
            labels[k] = "x"
        elif labels[k] == 'division':
            labels[k] = "/"
        elif labels[k] == 'IN0':
            labels[k] = "R"
    g = pgv.AGraph(directed=True, ranksep="2") #ranksep="2", nodesep="1.8", 
    g.add_nodes_from(nodes, color="cornflowerblue", style="filled")
    g.add_edges_from(edges, arrowhead="normal", color="gray11")
    g = g.reverse()
    g.layout(prog="twopi") #"dot"
    
    for i, n in enumerate(g.nodes()):
        if g.in_degree(n) == 0:
            node = g.get_node(n)
            node.attr["color"] = "chartreuse"
            node.attr["fillcolor"] = "chartreuse"
        elif labels[i] in ["root2", "root3", "root4"]:
            node = g.get_node(n)
            node.attr["color"] = "darkorange1"
            node.attr["fillcolor"] = "darkorange1"
        elif labels[i] == "=":
            node = g.get_node(n)
            node.attr["color"] = "darkorchid2"
            node.attr["fillcolor"] = "darkorchid2"
        if labels[i] == "R":
            node = g.get_node(n)
            node.attr["color"] = "gold"
            node.attr["fillcolor"] = "gold"

    for i in nodes:
        n = g.get_node(i)
        n.attr["shape"] = "box"
        n.attr["label"] = labels[i]

    if "end" in idx:
        g.draw(idx+"_tree.png")
        g.draw(idx+"_tree.pdf")
        g.close()
        return idx+"_tree.png"
    else:
        g.draw(idx+"_tree.pdf")
        g.close()

def plot(data, filename, title, ylabel):
    plt.figure(figsize=(12, 4))
    plt.grid(color='#F2F2F2', alpha=1, zorder=0)
    plt.plot(data, color='#087E8B', lw=3, zorder=5)
    plt.title(title, fontsize=17)
    plt.xlabel('Generations', fontsize=13)
    plt.xticks(fontsize=9)
    plt.ylabel(ylabel, fontsize=13)
    plt.yticks(fontsize=9)
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()
    return

def construct_plots(REPORT_PATH):
    df = pd.read_csv(REPORT_PATH+"/Output.csv")
    attributes_of_interest = {"avg":("Average_number_of_features","Number of features"), 
                              "avg.1":("Average_fitness_value","Fitness value"), 
                              "max.1":("Maximum_fitness_value","Fitness value"),
                              "time":("Generation_run_time","Time (seconds)"), 
                              "val_max":("Best_individual_validation_score","Validation performance"),
                              "val_mean":("Mean_validation_score_of_best_individuals","Validation performance"),
                              "avg.2":("Average_size_of_individuals","Number of nodes"),
                              "size":("Best_individual_size","Number of nodes")}

    PLOT_DIR = REPORT_PATH + "/plots/"
    # Generate plots
    for k,v in attributes_of_interest.items():
        plot(df[k], f'{PLOT_DIR}/{v}.png', v[0].replace("_", " "), v[1])

    # Construct data shown in document
    counter = 0
    pages_data = []
    temp = []
    # Get all plots
    files = os.listdir(REPORT_PATH + "/plots/")
    # Iterate over all created visualization
    for fname in files:
        # We want 3 per page
        if counter == 3:
            pages_data.append(temp)
            temp = []
            counter = 0

        temp.append(f'{REPORT_PATH}/plots/{fname}')
        counter += 1

def generate_report(REPORT_PATH, config):
    #Save plots
    construct_plots(REPORT_PATH)

    #Save parameter settings
    with open(REPORT_PATH+"/config.yml", "w") as file:
        yaml.dump(config, file, default_flow_style=False)