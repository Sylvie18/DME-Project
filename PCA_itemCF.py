import pandas as pd
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
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
    metriclist = ['PCA', 'rbf', 'cosine']
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
    for i, label in true.items():
        if label in pred[i]:
            rank = rank + pred[i].index(label) + 1

    return rank/len(true)

def eval(topres, allres, true):
    metric = {'precision': precision(topres, true),
              'meanrank': meanRank(allres, true)}

    return metric

def avgRes(allres):
    metriclist = ['PCA', 'rbf', 'cosine']
    res = {}

    for each in allres:
        for K in range(10, 90, 10):
            for metric in metriclist:
                res.setdefault('K='+str(K), {})
                res['K='+str(K)].setdefault(metric, {})
                res['K='+str(K)][metric].setdefault('precision', 0)
                res['K='+str(K)][metric].setdefault('meanrank', 0)

                res['K='+str(K)][metric]['precision'] += each['K='+str(K)][metric]['precision']
                res['K='+str(K)][metric]['meanrank'] += each['K='+str(K)][metric]['meanrank']

    for K in range(10, 90, 10):
        for metric in metriclist:
            res['K='+str(K)][metric]['precision'] = round(res['K='+str(K)][metric]['precision']/len(allres)*100, 2)
            res['K='+str(K)][metric]['meanrank'] = round(res['K='+str(K)][metric]['meanrank']/len(allres), 2)

    return res

def pcatrans(train_set):
    kernels = ['rbf', 'cosine']
    simlist = {}

    pca = PCA(n_components=train_set.shape[1])
    pca_train_set = pca.fit_transform(train_set.T)
    simlist['PCA'] = similarity(pd.DataFrame(pca_train_set.T))

    for kernel in kernels:
        kpca = KernelPCA(n_components=train_set.shape[1], kernel=kernel)
        kpca_train_set = kpca.fit_transform(train_set.T)
        simlist[kernel] = similarity(pd.DataFrame(kpca_train_set.T))

    return simlist


if __name__ == '__main__':
    file = 'preprocess_recipes.csv'
    allres = []

    # 5-fold evaluation
    for state in range(10, 60, 10):
        train_set, test_set = splitData(file, state)
        simlist = pcatrans(train_set)
        allres.append(completeRecipe(test_set, simlist, state))

    res = avgRes(allres)
    with open('pca_result.json', 'w') as f:
        json.dump(res, f, indent=2)
