import pandas as pd
import numpy as np
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA


# split dataset into train and test
def splitData(file):
    data = pd.read_csv(file)
    train_set, test_set = train_test_split(data, test_size=0.2, random_state=18)
    return train_set, test_set

# construct the inverted index of user -> item
def convertDict(data):
    res = {}
    for i in range(len(data)):
        line = data.iloc[[i]]
        line = line.loc[:, (line != 0).any(axis=0)]
        res.update(line.to_dict(orient='index'))

    return res

# calculate Cosine similarity
def simCos(Mi, Mij):
    res = {}
    for i, itemlist in Mij.items():
        res.setdefault(i, {})
        for j in itemlist.keys():
            res[i].setdefault(j, 0)
            res[i][j] = Mij[i][j] / sqrt(Mi[i]*Mi[j])

    return res

# calculate co-occurrence matrix of item -> item
# calculate similarity matrix
def similarity(dataset):
    data = convertDict(dataset)
    Mi = {}   # number of users who like item i
    Mij = {}  # number of users who like item i and item j

    for itemlist in data.values():
        for i in itemlist.keys():
            Mi.setdefault(i, 0)
            Mij.setdefault(i, {})
            Mi[i] += 1

            for j in itemlist.keys():
                if j != i:
                    Mij[i].setdefault(j, 0)
                    Mij[i][j] += 1

    return simCos(Mi, Mij)


if __name__ == '__main__':
    file = 'recipes.csv'
    train_set, test_set = splitData(file)
    kernels = ['rbf', 'cosine', 'sigmoid']
    simlist = {}

    pca = PCA(n_components=train_set.shape[1])
    pca_train_set = pca.fit_transform(train_set.T)
    simlist['PCA'] = similarity(pd.DataFrame(pca_train_set.T))

    for kernel in kernels:
        kpca = KernelPCA(n_components=train_set.shape[1], kernel=kernel)
        kpca_train_set = pca.fit_transform(train_set.T)
        simlist[kernel] = similarity(pd.DataFrame(kpca_train_set.T))
