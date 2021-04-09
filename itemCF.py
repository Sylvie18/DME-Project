import pandas as pd
import numpy as np
from math import sqrt
from sklearn.model_selection import train_test_split
import operator

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

# calculate Jaccard similarity
def simJaccard(Mi, Mij):
    res = {}
    for i, itemlist in Mij.items():
        res.setdefault(i, {})
        for j in itemlist.keys():
            res[i].setdefault(j, 0)
            res[i][j] = Mij[i][j] / (Mi[i]+Mi[j]-Mij[i][j])

    return res

# calculate Euclidean Distance similarity
def simEuclid(Mij):
    res = {}
    for i, itemlist in Mij.items():
        res.setdefault(i, {})
        for j in itemlist.keys():
            res[i].setdefault(j, 0)
            same = []

            for item in Mij[i]:
                if item in Mij[j]:
                    same.append(item)

            if len(same) == 0:
                res[i][j] = 0
            else:
                dis = sum([pow(Mij[i][item]-Mij[j][item], 2) for item in same])
                res[i][j] = 1 / (1+sqrt(dis))

    return res

# calculate Pointwise Mutual Information similarity
def simPMI(Mij):
    res = {}
    denominator = 0

    for itemlist in Mij.values():
        denominator += sum(itemlist.values())

    for i, itemlist in Mij.items():
        res.setdefault(i, {})
        for j in itemlist.keys():
            res[i].setdefault(j, 0)
            num = (Mij[i][j]/denominator) / ((sum(Mij[i].values())/denominator)*(sum(Mij[j].values())/denominator))
            if num != 0:
                res[i][j] = np.log2(num)
            else:
                res[i][j] = 0

    return res

# calculate co-occurrence matrix of item -> item
# calculate similarity matrix
def similarity(data):
    data = convertDict(data)
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

    simlist = {}
    simlist['Cosine'] = simCos(Mi, Mij)
    simlist['Jaccard'] = simJaccard(Mi, Mij)
    simlist['Euclidean'] = simEuclid(Mij)
    simlist['PMI'] = simPMI(Mij)

    return simlist

def recommandList(recipe, simlist, k, N=10):
    rank = {}
    for i, score in recipe.items():
        for j, sim in sorted(simlist['Cosine'][i].items(), key=operator.itemgetter(1), reverse=True):
            if j not in recipe.keys():
                rank.setdefault(j,0)
                rank[j] += float(score) * sim
                
    # print("---Recommandation---")
    # print(sorted(rank.items(), key=operator.itemgetter(1), reverse=True)[0:N])
    return sorted(rank.items(), key=operator.itemgetter(1), reverse=True)[0:N]
    


if __name__ == '__main__':
    file = 'recipes.csv'
    train_set, test_set = splitData(file)
    simlist = similarity(train_set)
    # recomList = recommandList(a, simlist)