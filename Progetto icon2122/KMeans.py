# -*- coding: utf-8 -*-

import numpy as np, pandas as pd, seaborn as sn
import pickle
from sklearn.cluster import KMeans  
import seaborn as sns; sns.set()  #per lo stile dei plot
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from inspect import signature

def KMEANS(X,y):
    
    kmeans = KMeans(n_clusters = 2, init = 'k-means++', max_iter = 2, n_init = 9, random_state = 0)
    y_kmeans = kmeans.fit_predict(X)

    centroids = kmeans.cluster_centers_
    print("\nEtichette:")  
    print(kmeans.labels_)
    average_precision = average_precision_score(y, y_kmeans)
    precision, recall, _ = precision_recall_curve(y, y_kmeans)


    # In matplotlib < 1.5, plt.fill_between does not have a 'step' argument
    step_kwargs = ({'step': 'post'}
               if 'step' in signature(plt.fill_between).parameters
               else {})
    plt.step(recall, precision, color='b', alpha=0.2,
         where='post')
    plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))


    accuracy = accuracy_score(y_kmeans, y)
    print ('\nClasification report:\n',classification_report(y, y_kmeans))


    confusion_Matrix = confusion_matrix(y, y_kmeans)
    df_cm = pd.DataFrame(confusion_Matrix, index = [i for i in "01"], columns = [i for i in "01"])
    plt.figure(figsize = (10,7))

    sn.heatmap(df_cm, annot=True)
    plt.show()
    
    print('accuracy, average precision are:', accuracy, average_precision)
    
    #salvo il modello su disco per uso futuro
    pickle.dump(kmeans, open("MODEL_kmeans.sav", 'wb'))
