import pandas as pd
import cv2
import numpy as np
import sys
sys.path.append("..")
from .primitives import HyperspectralImg, ReflectanceSpectra
from algorithms import SPXY
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

def read_sediment_data(attribute="Porosity"):
    df = pd.read_excel("Sediment_reference.xlsx")
    x_train, y_train, x_test, y_test = [], [], [], []
    X, X_values, y = [], [], []
    for i, row in df.iterrows():
        if row[attribute] != -1:
            HSI_file = "./Dataset/Hyperspectral/"+row["Filename"]+"_"+row["Suffix"]+"_data.npy"
            ROI_file = "./Dataset/ROI/"+row["Filename"]+"_"+row["Suffix"]+"ROI.png"
            ROI = cv2.imread(ROI_file, 0)

            HS_image = np.load(HSI_file)
            
            # #Find shells and specular highlights
            # _, specular = cv2.threshold(HS_image[:,:,160], 0.3, 255, cv2.THRESH_BINARY)
            # specular = cv2.dilate(specular, np.ones((3,3),np.uint8))
            # spec_positions = np.where(specular)
            # new_ROI = np.zeros(ROI.shape, dtype=bool)
            # new_ROI[ROI > 0] = True
            # new_ROI[spec_positions] = False
            # ROI = new_ROI

            spectral_response = (HS_image.sum(axis=(0, 1))/(np.sum(ROI > 1)))
            spectral_response = (spectral_response-np.mean(spectral_response,keepdims=True))/np.std(spectral_response,keepdims=True)
            spectral_response  = savgol_filter(spectral_response, 5, 2, mode='nearest')

            spectra = ReflectanceSpectra(spectral_response)
            ROI = ROI.astype(bool)
            X_values += [spectra]
            X += [HyperspectralImg(HS_image, ROI, spectra)]
            y += [[float(row["Porosity"]), float(row["Organic Matter"])]]

    y = np.asarray(y).astype(np.float64)

    x_train, x_test, y_train, y_test = SPXY.split(X, X_values, y, train_size=0.6666)

    if attribute == "Porosity":
        y_test = np.asarray([y_test[i][0] for i in range(len(y_test))])
        y_train = np.asarray([y_train[i][0] for i in range(len(y_train))])
    else:
        y_test = np.asarray([y_test[i][1] for i in range(len(y_test))])
        y_train = np.asarray([y_train[i][1] for i in range(len(y_train))])

    return x_train, y_train, x_test, y_test