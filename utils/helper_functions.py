import numpy as np
from sklearn import svm
import glob
import os
import utils.dataloader_sediment as sediment_loader


def update_hof(offspring_for_va, halloffame):
    """Update the hall of fame with the generated individuals
    """
    if halloffame is not None:
        halloffame.update(offspring_for_va)
    return halloffame

def evalTesting(toolbox, individual, x_train, y_train, x_test, y_test):
    """Gives the R^2 for the testing set
    individual : a GP tree/individual
    x_train : training image
    y_train : ground truth segmentation
    """
    func = toolbox.compile(expr=individual)
    X = []
    for c in range(len(y_train)):
        X += [func(x_train[c])]
    for c in range(len(y_test)):
        X += [func(x_test[c])]

    X = np.asarray(X)
    X = np.nan_to_num(X, nan=0.0)
    # Standardise the input (X)
    X = np.divide(X-np.mean(X, axis=0, keepdims=True), np.std(X, axis=0, keepdims=True), out=np.zeros_like(X), where=np.std(X, axis=0, keepdims=True)!=0)
    
    regr = svm.SVR(kernel='linear')
    regr.fit(X[:len(y_train)], y_train)

    res_training = regr.score(X[:len(y_train)], y_train)
    res_testing = regr.score(X[len(y_train):], y_test)

    return {"R^2 Testing": res_testing, "R^2 Training": res_training}


def import_data(image_based = False, attribute="Porosity"):
    """Import hyperspectral data, preprocess it into mean reflectance spectra and split 
    the data.
    Returns : training x,y data and test x,y data as np arrays
    """
    x_train, y_train, x_test, y_test = sediment_loader.read_sediment_data(attribute)

    if not image_based:
        for i in range(len(x_train)):
            x_train[i] = x_train[i].spectra
        for i in range(len(x_test)):
            x_test[i] = x_test[i].spectra

    print("Training/test set sizes:", len(x_train), len(x_test))

    return x_train, y_train, x_test, y_test

def get_reporting_path():
    #Get name of new report
    files = sorted(glob.glob('.\Reports\exp*'))
    if len(files) == 0:
        exp_name = "exp_"+str(1).zfill(4)
    else:
        exp_name = "exp_" + str(int(files[-1].split("_")[-1])+1).zfill(4)

    file_path = "./Reports/" + exp_name
    os.mkdir(file_path)
    os.mkdir(file_path + "/plots/")
    os.mkdir(file_path + "/trees/")
    return file_path

#Functions for generating fitness outputs.
#The default variable is the default fitness value on failure
def rounded_mean(default, x):
    arr = np.asarray(x)
    return np.round(np.mean(arr[arr!=default]), 4)

def rounded_std(default, x):
    arr = np.asarray(x)
    return np.round(np.std(arr[arr!=default]), 4)

def rounded_max(default, x):
    #return np.round(np.max(x[x!=default]), 4)
    return np.round(np.max(x), 4)

def rounded_min(default, x):
    #return np.round(np.min(x[x!=default]), 4)
    return np.round(np.min(x), 4)
