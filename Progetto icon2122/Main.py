# -*- coding: utf-8 -*-

import numpy as np, pandas as pd
import pickle
import KNearestNeighbors, MultinomialNaiveBayes, RandomForest, SupportVectorMachine, KMeans
import ontology_manager
import CSP_AnalysisPrenotation


def main():
    print("Preparazione dati...")
    #Preparazione dati
    feature = ["class", "age", "menopause", "tumor-size", "inv-nodes", "node-caps", "deg-malig", "bladder", "bladder-quad", "irradiat"]
    feature_dummied = ["age", "menopause", "tumor-size", "inv-nodes", "node-caps", "bladder", "bladder-quad", "irradiat"]
    dataset = pd.read_csv("breast-cancer.csv", sep=",", names=feature, dtype={'class':object, 'age':object, 'menopause':object, 'tumor-size':object, 'inv-nodes':object, 'node-caps':object, 'deg-malig':np.int32, 'bladder': object, 'bladder-quad':object, 'irradiat':object})
    data_dummies = pd.get_dummies(dataset, columns=feature_dummied)
    data_dummies = data_dummies.drop(["class"], axis=1)

    x = data_dummies
    y = pd.get_dummies(dataset["class"], columns=["class"])
    y = y["recurrence-events"]
    print("...completata.")
    input("Press Enter to continue...")
    
    #Classificazione supervisionata
    print("ALGORITMO: K-Nearest Neighbors")
    KNearestNeighbors.KNN(x,y)
    input("Press Enter to continue...")
    
    print("ALGORITMO: Random Forest")
    RandomForest.RF(x,y)
    input("Press Enter to continue...")
    
    print("ALGORITMO: Multinomial Naive Bayes")
    MultinomialNaiveBayes.MNB(x,y)
    input("Press Enter to continue...")
    
    print("ALGORITMO: Support Vector Machine")
    SupportVectorMachine.SVM(x,y)
    input("Press Enter to continue...")
    
    #Classificazione non supervisionata
    print("ALGORITMO: K-Means")
    KMeans.KMEANS(x,y)
    
    #Caricamento modello addestrato
    #filename = "knn_model.sav"
    #loaded_model = pickle.load(open(filename, 'rb'))
    #result = loaded_model.score(X_test, Y_test)
    #print(result)
    
    #Stampa il contenuto della ontologia
    print("\n\nONTOLOGIA DI DOMINIO\n")
    ontology_manager.ontology_analyzer()
    
    #Prenota una visita istologica
    CSP_AnalysisPrenotation.lab_booking()
    