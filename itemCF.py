import pandas as pd
import numpy as np
from math import sqrt
from sklearn.model_selection import train_test_split
import operator
import random
import json

# split dataset into train and test
def splitData(file, state):
    data = pd.read_csv(file)
    train_set, test_set = train_test_split(data, test_size=0.2, random_state=state)
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

            if len(same) != 0:
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

def make_missingIngs_set(data, state):
    misset = convertDict(data)
    misIngs = {}
    random.seed(state)

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

def completeRecipe(test_set, simlist, state):
    misset, misIngs = make_missingIngs_set(test_set, state)
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

    return hit/len(true)

def meanRank(pred, true):
    rank = 0
    num = 0
    for i, label in true.items():
        if label in pred[i]:
            rank = rank + pred[i].index(label) + 1
            num += 1

    if num != 0:
        return rank/num
    return 0

def eval(topres, allres, true):
    metric = {'precision': precision(topres, true),
              'meanrank': meanRank(allres, true)}

    return metric

def avgRes(allres):
    metriclist = ['Cosine', 'Jaccard', 'Euclidean', 'PMI']
    res = {}

    for each in allres:
        for K in range(10, 90, 10):
            kvalue = 'K='+str(K)
            for metric in metriclist:
                res.setdefault(kvalue, {})
                res[kvalue].setdefault(metric, {})
                res[kvalue][metric].setdefault('precision', 0)
                res[kvalue][metric].setdefault('meanrank', 0)

                res[kvalue][metric]['precision'] += each[kvalue][metric]['precision']
                res[kvalue][metric]['meanrank'] += each[kvalue][metric]['meanrank']

    for K in range(10, 90, 10):
        kvalue = 'K='+str(K)
        for metric in metriclist:
            res[kvalue][metric]['precision'] = round(res[kvalue][metric]['precision']/len(allres)*100, 2)
            res[kvalue][metric]['meanrank'] = round(res[kvalue][metric]['meanrank']/len(allres), 2)

    return res


if __name__ == '__main__':
    file = 'preprocess_recipes.csv'
    allres = []

    # 5-fold evaluation
    for state in range(10, 60, 10):
        train_set, test_set = splitData(file, state)
        simlist = similarity(train_set)
        allres.append(completeRecipe(test_set, simlist, state))

    res = avgRes(allres)
    with open('result.json', 'w') as f:
        json.dump(res, f, indent=2)
