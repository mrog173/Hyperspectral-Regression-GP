import cv2
import numpy as np
import os
from scipy.signal import savgol_filter
import pandas as pd
from utils.primitives import *
from utils.initialise_primitives import spectra_based, image_based
import matplotlib.pyplot as plt
from deap import base, creator, gp, tools
from sklearn import metrics, svm
import matplotlib.patches as patches

def import_data(image_based = False, attribute="Porosity"):
    """Import hyperspectral data, preprocess it into mean reflectance spectra and split 
    the data.
    Returns : training x,y data and test x,y data as np arrays
    """
    X, X_values, names, training_idx, training_vals = read_all_sediment_data()

    if not image_based:
        return X_values, names, training_idx, training_vals

    return X, names, training_idx, training_vals

def read_all_sediment_data():
    X, X_values, names, training_idx, training_vals = [], [], [], [], []
    df = pd.read_excel("Sediment_reference.xlsx")
    for i, file in enumerate(os.listdir("./Dataset/Hyperspectral/")):
        if (df['Filename'] == file[:-12]).any():
            if ((df['Suffix'] == file[-11:-9]) & (df['Filename'] == file[:-12])).any():
                row = df.loc[(df['Suffix'] == file[-11:-9]) & (df['Filename'] == file[:-12])]
                if float(row["Organic Matter"]) != -1:
                    training_idx += [len(X)]
                    training_vals += [float(row["Organic Matter"])]

            HSI_file = "./Dataset/Hyperspectral/"+file
            ROI_file = "./Dataset/ROI/"+file[:-9]+"ROI.png"
            ROI = cv2.imread(ROI_file, 0)

            names += [file[:-9]]

            HS_image = np.load(HSI_file)

            spectral_response = (HS_image.sum(axis=(0, 1))/(np.sum(ROI > 1)))
            spectral_response = (spectral_response-np.mean(spectral_response,keepdims=True))/np.std(spectral_response,keepdims=True)
            spectral_response  = savgol_filter(spectral_response, 5, 2, mode='nearest')

            spectra = ReflectanceSpectra(spectral_response)
            ROI = ROI.astype(bool)
            X_values += [spectra]
            X += [HyperspectralImg(HS_image, ROI, spectra)]

    print(len(X), len(X_values), len(names), len(training_idx), len(training_vals))
    return X, X_values, names, training_idx, training_vals

def split(X_vals, y_vals, training_idx, train_size = 0.6666):
    """
    Repeats SPXY split to get the training set indices
    """
    X = np.asarray(X_vals)
    y = np.asarray(y_vals).reshape(-1,1)
    
    n = len(X)
    k = n*train_size
    
    x_dist = metrics.pairwise_distances(X, metric='euclidean', n_jobs=-1)
    x_dist = x_dist/np.max(x_dist)
    y_dist = metrics.pairwise_distances(y, metric='euclidean', n_jobs=-1)
    y_dist = y_dist/np.max(y_dist)

    dist = x_dist + y_dist

    full_set = set(list(range(n)))
    # Select two samples with largest distance
    idx_0, idx_1 = np.unravel_index(np.argmax(dist, axis=None), dist.shape)
    selected = set([idx_0, idx_1])
    k -= 2
    m_j = idx_1

    # Iteratively select samples with the largest minimum distance
    while k > 0 and len(selected) < n:
        minimum_dist = 0
        for j in range(n):
            if j not in selected:
                j_minimum = min([dist[j][i] for i in selected])
                if j_minimum > minimum_dist:
                    m_j = j
                    minimum_dist = j_minimum
        selected.add(m_j)
        k -= 1

    y_train = [y[s] for s in selected]
    training_set_idx = [training_idx[s] for s in selected]

    # Return training and testing set
    return training_set_idx, y_train


def main():
    image_based = False
    attribute = "Organic Matter"
    
    res, names, training_idx, training_vals = import_data(image_based=False, attribute=attribute)
    
    if image_based:
        primitive_set = image_based(res[0].shape[2])
    else:
        primitive_set = spectra_based(res[0].shape[0])

    res = np.asarray(res)
    training_set_idx, y_train = split(res[training_idx], training_vals, training_idx, train_size=0.6666)
    print(len(training_set_idx), len(y_train))

    # Need to initialise the GP toolbox to compile the function string
    toolbox = base.Toolbox()
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)
    toolbox.register("expr", gp.genHalfAndHalf, pset=primitive_set, min_=1, max_=20)
    toolbox.register("compile", gp.compile, pset=primitive_set)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)

    ###Change this to the string you want to apply
    func_str = "root3(convert_to_feature(division(median_interval_selection(IN0, 164, 3), median_interval_selection(IN0, 155, 9))), root4(convert_to_feature(mean_interval_selection(IN0, 168, 1)), root2(convert_to_feature(subtraction(mean_interval_selection(IN0, 197, 1), median_interval_selection(IN0, 192, 3))), root4(convert_to_feature(mean_interval_selection(IN0, 182, 11)), root4(convert_to_feature(mean_interval_selection(IN0, 182, 11)), convert_to_feature(mean_interval_selection(IN0, 168, 3)), convert_to_feature(mean_interval_selection(IN0, 60, 11)), convert_to_feature(mean_interval_selection(IN0, 189, 1))), convert_to_feature(median_interval_selection(IN0, 13, 9)), root4(convert_to_feature(mean_interval_selection(IN0, 182, 11)), convert_to_feature(addition(median_interval_selection(IN0, 193, 1), mean_interval_selection(IN0, 140, 13))), convert_to_feature(mean_interval_selection(IN0, 60, 11)), convert_to_feature(median_interval_selection(IN0, 129, 7))))), convert_to_feature(mean_interval_selection(IN0, 182, 11)), convert_to_feature(mean_interval_selection(IN0, 189, 7))), root4(convert_to_feature(mean_interval_selection(IN0, 114, 11)), convert_to_feature(mean_interval_selection(IN0, 129, 1)), convert_to_feature(median_interval_selection(IN0, 198, 9)), convert_to_feature(division(median_interval_selection(IN0, 164, 9), median_interval_selection(IN0, 155, 9)))))"
    #func_str = "root3(root2(root4(convert_to_feature(mean_interval_selection(IN0, 94, 13)), convert_to_feature(mean_interval_selection(IN0, 41, 9)), root4(convert_to_feature(mean_interval_selection(IN0, 143, 9)), convert_to_feature(median_interval_selection(IN0, 76, 11)), convert_to_feature(mean_interval_selection(IN0, 119, 3)), root4(convert_to_feature(median_interval_selection(IN0, 6, 5)), convert_to_feature(median_interval_selection(IN0, 17, 1)), convert_to_feature(mean_interval_selection(IN0, 29, 11)), convert_to_feature(median_interval_selection(IN0, 194, 5)))), root4(root4(convert_to_feature(median_interval_selection(IN0, 25, 1)), convert_to_feature(median_interval_selection(IN0, 73, 9)), convert_to_feature(median_interval_selection(IN0, 65, 11)), convert_to_feature(mean_interval_selection(IN0, 151, 3))), root4(convert_to_feature(median_interval_selection(IN0, 6, 5)), convert_to_feature(median_interval_selection(IN0, 17, 1)), convert_to_feature(median_interval_selection(IN0, 89, 11)), convert_to_feature(mean_interval_selection(IN0, 187, 1))), convert_to_feature(addition(mean_interval_selection(IN0, 163, 9), mean_interval_selection(IN0, 163, 9))), convert_to_feature(median_interval_selection(IN0, 129, 13)))), root4(convert_to_feature(median_interval_selection(IN0, 185, 1)), convert_to_feature(median_interval_selection(IN0, 165, 7)), convert_to_feature(median_interval_selection(IN0, 103, 3)), root4(convert_to_feature(mean_interval_selection(IN0, 94, 13)), root4(convert_to_feature(mean_interval_selection(IN0, 98, 1)), convert_to_feature(median_interval_selection(IN0, 25, 1)), convert_to_feature(median_interval_selection(IN0, 129, 11)), convert_to_feature(median_interval_selection(IN0, 68, 3))), root4(convert_to_feature(mean_interval_selection(IN0, 143, 9)), convert_to_feature(division(mean_interval_selection(IN0, 181, 13), mean_interval_selection(IN0, 6, 7))), convert_to_feature(mean_interval_selection(IN0, 119, 13)), convert_to_feature(mean_interval_selection(IN0, 14, 3))), root4(root4(convert_to_feature(median_interval_selection(IN0, 25, 1)), convert_to_feature(median_interval_selection(IN0, 73, 13)), convert_to_feature(median_interval_selection(IN0, 65, 11)), convert_to_feature(mean_interval_selection(IN0, 151, 3))), convert_to_feature(median_interval_selection(IN0, 39, 7)), convert_to_feature(addition(mean_interval_selection(IN0, 64, 11), mean_interval_selection(IN0, 163, 9))), convert_to_feature(median_interval_selection(IN0, 129, 13)))))), root4(root4(convert_to_feature(median_interval_selection(IN0, 103, 3)), root4(convert_to_feature(mean_interval_selection(IN0, 94, 13)), convert_to_feature(median_interval_selection(IN0, 33, 3)), root4(convert_to_feature(mean_interval_selection(IN0, 143, 9)), convert_to_feature(mean_interval_selection(IN0, 86, 1)), convert_to_feature(mean_interval_selection(IN0, 76, 3)), convert_to_feature(mean_interval_selection(IN0, 14, 3))), root4(root4(convert_to_feature(median_interval_selection(IN0, 25, 1)), convert_to_feature(median_interval_selection(IN0, 73, 9)), convert_to_feature(median_interval_selection(IN0, 65, 11)), convert_to_feature(mean_interval_selection(IN0, 151, 3))), root4(convert_to_feature(median_interval_selection(IN0, 6, 5)), convert_to_feature(median_interval_selection(IN0, 181, 1)), convert_to_feature(mean_interval_selection(IN0, 29, 11)), convert_to_feature(median_interval_selection(IN0, 86, 13))), convert_to_feature(addition(mean_interval_selection(IN0, 64, 11), mean_interval_selection(IN0, 163, 9))), convert_to_feature(mean_interval_selection(IN0, 192, 3)))), convert_to_feature(mean_interval_selection(IN0, 43, 3)), convert_to_feature(median_interval_selection(IN0, 68, 3))), convert_to_feature(division(median_interval_selection(IN0, 181, 1), mean_interval_selection(IN0, 6, 7))), root2(root4(root4(convert_to_feature(mean_interval_selection(IN0, 143, 9)), convert_to_feature(mean_interval_selection(IN0, 86, 1)), convert_to_feature(mean_interval_selection(IN0, 119, 13)), convert_to_feature(mean_interval_selection(IN0, 14, 3))), convert_to_feature(median_interval_selection(IN0, 33, 3)), root4(root4(convert_to_feature(median_interval_selection(IN0, 25, 1)), convert_to_feature(median_interval_selection(IN0, 73, 13)), convert_to_feature(median_interval_selection(IN0, 65, 11)), convert_to_feature(median_interval_selection(IN0, 6, 5))), root4(convert_to_feature(median_interval_selection(IN0, 6, 5)), convert_to_feature(median_interval_selection(IN0, 17, 1)), convert_to_feature(mean_interval_selection(IN0, 29, 11)), convert_to_feature(median_interval_selection(IN0, 194, 5))), convert_to_feature(addition(mean_interval_selection(IN0, 64, 11), mean_interval_selection(IN0, 163, 9))), convert_to_feature(median_interval_selection(IN0, 129, 1))), root4(root4(convert_to_feature(median_interval_selection(IN0, 196, 1)), convert_to_feature(median_interval_selection(IN0, 73, 13)), convert_to_feature(median_interval_selection(IN0, 65, 11)), convert_to_feature(mean_interval_selection(IN0, 151, 3))), root4(convert_to_feature(median_interval_selection(IN0, 6, 5)), convert_to_feature(median_interval_selection(IN0, 17, 1)), convert_to_feature(mean_interval_selection(IN0, 29, 11)), convert_to_feature(median_interval_selection(IN0, 194, 5))), convert_to_feature(addition(mean_interval_selection(IN0, 64, 11), mean_interval_selection(IN0, 163, 9))), convert_to_feature(median_interval_selection(IN0, 129, 1)))), root4(convert_to_feature(median_interval_selection(IN0, 185, 1)), convert_to_feature(median_interval_selection(IN0, 165, 7)), convert_to_feature(median_interval_selection(IN0, 103, 3)), convert_to_feature(mean_interval_selection(IN0, 187, 5)))), convert_to_feature(multiplication(median_interval_selection(IN0, 9, 9), median_interval_selection(IN0, 191, 5)))), root4(convert_to_feature(mean_interval_selection(IN0, 32, 13)), convert_to_feature(median_interval_selection(IN0, 135, 11)), root4(convert_to_feature(mean_interval_selection(IN0, 26, 5)), convert_to_feature(mean_interval_selection(IN0, 152, 11)), convert_to_feature(median_interval_selection(IN0, 89, 11)), convert_to_feature(median_interval_selection(IN0, 196, 13))), root2(convert_to_feature(median_interval_selection(IN0, 129, 1)), convert_to_feature(median_interval_selection(IN0, 196, 11)))))"
    #DEAP has an issue where it will not accept a int when it is expecting a subclass
    #e.g., a WavelengthBand. To prevent redefining the primitive set, here edit the
    #primitives and replace their subclass with their superclasses where relevant.
    replacements = {
        WavelengthBand:int,
        WindowWidth:int,
        Sigma:float,
        KernelSize:int,
        GaborTheta:float,
        GaborFreq:float,
        GLCMTheta:float,
        GLCMDistance:float
    }

    for p in primitive_set.primitives:
        for f in primitive_set.primitives[p]:
            for i, g in enumerate(f.args):
                if g in replacements:
                    f.args[i] = replacements[g]
    
    func = creator.Individual.from_string(func_str, pset=primitive_set)
    func = toolbox.compile(expr=func)
    
    X = []
    for c in range(len(res)):
        X += [func(res[c])]

    X = np.nan_to_num(X, nan=0.0)

    # Standardise the input (X)
    X = np.divide(X-np.mean(X, axis=0, keepdims=True), np.std(X, axis=0, keepdims=True), out=np.zeros_like(X), where=np.std(X, axis=0, keepdims=True)!=0)
    
    regr = svm.SVR(kernel='linear')
    regr.fit(X[training_set_idx], y_train)

    result = regr.predict(X)
    
    with open("Application_results.csv","w") as out:
        out.write("name, prediction, testing\n")
        for idx, obs in enumerate(X):
            training = True if idx in training_set_idx else False
            truth = "-"
            if idx in training_idx:
                truth = training_vals[training_idx.index(idx)]
            out.write(names[idx]+","+str(result[idx])+","+str(truth)+","+str(training)+"\n")

    width = 5
    height = 6
    df = pd.read_csv("Application_results.csv")
    fig, ax = plt.subplots(5,6)

    row_idx = ["A", "B", "C", "D", "E"]

    plots = dict()
    trained_samples = dict()

    for i, row in df.iterrows():
        observation = row.name[:-3]
        x = row_idx.index(row.name[-2])
        y = int(row.name[-1])
        
        if not observation in plots:
            plots[observation] = np.zeros((5,5))
            trained_samples[observation] = []
        plots[observation][x,y] = row['name']
        if row[" prediction"] != "-" and row[" testing"]:
            trained_samples[observation] += [(x,y)]
        

    for i,(k,v) in enumerate(plots.items()):
        v = cv2.resize(v, (50,50), interpolation=cv2.INTER_NEAREST)
        im = ax.flat[i].imshow(v, vmax=max(df['name']), vmin=min(df['name']))
        ax.flat[i].axis('off')

        for r in trained_samples[k]:
            rect = patches.Rectangle((r[1]*10,r[0]*10), 10, 10, linewidth=1, edgecolor='r', linestyle="--", facecolor='none', label='Training data')
            ax.flat[i].add_patch(rect)

    #Fix legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=[-2, 0], loc="upper center")

    cbaxes = fig.add_axes([0.85, 0.1, 0.03, 0.8]) 
    cb = fig.colorbar(im, ax=ax.ravel().tolist(), drawedges=False, cax = cbaxes)
    cb.outline.set_visible(False)
    cb.set_label('Predicted organic matter content', rotation=270, fontsize = 15, labelpad=20)

    plt.subplots_adjust(left=0.05, bottom=0.08, right=0.82, top=0.95, wspace=0.03, hspace=0.03)
    plt.show()
    

main()