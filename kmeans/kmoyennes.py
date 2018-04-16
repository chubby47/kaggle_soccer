# -*- coding: utf-8 -*-

# ------------------------------------------------
# package IADS2018
# UE 3I026 "IA et Data Science" -- 2017-2018
#
# Module kmoyennes.py:
# Fonctions pour le clustering
# ------------------------------------------------

# Importations nécessaires pour l'ensemble des fonctions de ce fichier:
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import random
from sklearn.decomposition import PCA

# Normalisation des données :

def normalise(value, min, max):
    return float((value - min)) / float(max - min)
    

vfunc = np.vectorize(normalise)

def normalisation(dataframe):
    """
    normalise un dataframe
    """
    df = dataframe.copy()
    for e in df.columns:
        df[e] = vfunc(df[e], df[e].min(), df[e].max())
    return df

# -------
# Fonctions distances

def dist_euclidienne_vect(v1, v2):
    return np.linalg.norm(v2 - v1)

def dist_manhattan_vect(v1, v2):
    return sum(map(abs, v2 - v1))

def dist_vect(v1, v2, heuristique="euclidian"):
    return dist_euclidienne_vect(v1, v2) if heuristique=="euclidian" else dist_manhattan_vect(v1, v2)

# -------
# Calculs de centroïdes :
def centroide(matrix):
    return (np.sum(matrix, axis=0) / len(matrix)).to_frame().transpose()

# -------
# Inertie des clusters :
def inertie_cluster(cluster):
    cen = centroide(cluster).iloc[0]
    sum = 0
    for i in range(len(cluster)):
        sum += dist_vect(cluster.iloc[i], cen)**2
    return sum
# -------
# Algorithmes des K-means :
def initialisation( k, baseApprentissage):
    return baseApprentissage.iloc[np.random.choice(len(baseApprentissage), k, replace=False)]

# -------
def plus_proche(exemple, centroids):
    minIndex = 0
    minimum = 100000
    for i in range(len(centroids)):
        if dist_vect(exemple,centroids.iloc[i]) < minimum:
            minimum = dist_vect(exemple,centroids.iloc[i])
            minIndex = i
    return minIndex
# -------
def affecte_cluster(base, centroids):
    mat = dict()
    for i in range(len(base)):
        pp = plus_proche(base.iloc[i], centroids)
        if pp in mat:
            mat[pp].append(i)
        else:
            mat[pp] = [i]
    return mat
        
# -------
def nouveaux_centroides(base, affectation):
    df = pd.DataFrame()
    for liste in affectation.values():
        d = centroide(base.iloc[liste])
        df = pd.concat([d, df])
    return df
# -------
def inertie_globale(base, dictAffect):
    sum = 0
    for cluster in dictAffect.values():
        sum += inertie_cluster(base.iloc[cluster])
    return sum
# ************************* Recopier ici la fonction kmoyennes()
def kmoyennes( k, base, epsilon, iter_max):
    inertie = 1000
    last_inertie = 0
    i = 0
    cen = initialisation(k, base)
    affect = affecte_cluster(base, cen)
    inertie = inertie_globale(base, affect)
    
    while i < iter_max and abs(inertie - last_inertie) >= epsilon:
        cen = nouveaux_centroides(base, affect)
        affect = affecte_cluster(base, cen)
        last_inertie = inertie
        inertie = inertie_globale(base, affect)
        i += 1
    
    return cen, affect
# -------
# Affichage :
colors = ['red', 'blue', 'green', 'black', 'yellow', 'pink']

def affiche_resultat(DataFnorm, les_centres, affect):
    plt.scatter(les_centres['X'],les_centres['Y'],color='r',marker='x')
    c = 0
    for cluster in affect.values():
        plt.scatter(DataFnorm.iloc[cluster]['X'],DataFnorm.iloc[cluster]['Y'],color=colors[c])
        c += 1
    plt.show()
# -------

def visualiser_clusters(dataf, clusters):
    colors = ['red', 'green', 'blue', 'black', 'yellow', 'grey']
    a, b = dataf.shape
    nb_clusters = len(clusters)
    labels = [0] * len(dataf)
    for c, liste in clusters.items():
        for i in liste:
            labels[i] = c
    #utilisation de l'algorithme PCA : 
    pca = PCA(n_components=2).fit(dataf)
    #reduction de dimension
    pca_2d = pca.transform(dataf)
    x = []
    y = []
    for row in pca_2d:
        x.append(row[0])
        y.append(row[1])
    for i in range(len(x)):
        plt.scatter(x[i], y[i], c=colors[labels[i]])
    plt.title(str(nb_clusters)+" categories d'arrondissements de Paris en fonction de leur production de dechets")
    plt.show()