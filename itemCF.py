import pandas as pd
import numpy as np
from math import sqrt
from sklearn.model_selection import train_test_split
import operator
import random
import json

# split dataset into train and test
def splitData(data):
    # data = pd.read_csv(file)
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

def make_missingIngs_set(data):
    misset = convertDict(data)
    misIngs = {}

    random.seed(18)

    for i, ingList in misset.items():
        misIng = random.choice(list(ingList))
        misIngs[i] = misIng
        del (misset[i][misIng])

    return misset, misIngs

def recommendList(recipes, simdict, K, N = 10):
    rank = {}
    topres = {}
    allres = {}

    for i, recipe in recipes.items():
        rank.setdefault(i, {})
        for j, score in recipe.items():
            if j in simdict:
                for k, sim in sorted(simdict[j].items(), key=operator.itemgetter(1), reverse=True)[0:K]:
                    if k not in recipe.keys():
                        rank[i].setdefault(k, 0)
                        rank[i][k] += float(score) * sim

        rank[i] = sorted(rank[i].items(), key=operator.itemgetter(1), reverse=True)
        topres[i] = [each[0] for each in rank[i][0:N]]
        allres[i] = [each[0] for each in rank[i]]

    return topres, allres

def completeRecipe(test_set, simlist):
    misset, misIngs = make_missingIngs_set(test_set)
    metriclist = ['Cosine', 'Jaccard', 'Euclidean', 'PMI']
    res = {}

    for K in range(10, 90, 10):
        subres = {}
        for metric in metriclist:
            topres, allres = recommendList(misset, simlist[metric], K)
            subres[metric] = eval(topres, allres, misIngs)
        res['K='+str(K)] = subres

    return res

def precision(pred, true):
    hit = 0
    for i, label in true.items():
        if label in pred[i]:
            hit += 1

    return round(hit/len(true)*100, 2)

def meanRank(pred, true):
    rank = 0
    for i, label in true.items():
        if label in pred[i]:
            rank = rank + pred[i].index(label) + 1

    return round(rank/len(true), 2)

def eval(topres, allres, true):
    metric = {'precision': precision(topres, true),
              'meanrank': meanRank(allres, true)}

    return metric


if __name__ == '__main__':
    file = 'deletedquotes.csv'
    data = pd.read_csv(file)

    for index, row in data.iteritems():
        if row.sum() < 10:
            data = data.drop(index, axis=1)

    train_set, test_set = splitData(data)
    simlist = similarity(train_set)
    result = completeRecipe(test_set, simlist)

    with open('result.json', 'w') as f:
        json.dump(result, f, indent=2)
