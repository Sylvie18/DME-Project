import pandas as pd
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
import operator
import random
import json
import mca
import prince
from sklearn import manifold

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
            res[i][j] = abs(Mij[i][j]) / sqrt(abs(Mi[i])*abs(Mi[j]))

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
            Mi[i] += itemlist[i]

            for j in itemlist.keys():
                if j != i:
                    Mij[i].setdefault(j, 0)
                    Mij[i][j] += itemlist[j]

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
    metriclist = ['PCA', 'MCA', 'TSNE']
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
        return rank / num
    return 0

def eval(topres, allres, true):
    metric = {'precision': precision(topres, true),
              'meanrank': meanRank(allres, true)}

    return metric

def avgRes(allres):
    metriclist = ['PCA', 'MCA', 'TSNE']
    res = {}

    for each in allres:
        for K in range(10, 90, 10):
            kvalue = 'K=' + str(K)
            for metric in metriclist:
                res.setdefault(kvalue, {})
                res[kvalue].setdefault(metric, {})
                res[kvalue][metric].setdefault('precision', 0)
                res[kvalue][metric].setdefault('meanrank', 0)

                res[kvalue][metric]['precision'] += each[kvalue][metric]['precision']
                res[kvalue][metric]['meanrank'] += each[kvalue][metric]['meanrank']

    for K in range(10, 90, 10):
        kvalue = 'K=' + str(K)
        for metric in metriclist:
            res[kvalue][metric]['precision'] = round(res[kvalue][metric]['precision'] / len(allres) * 100, 2)
            res[kvalue][metric]['meanrank'] = round(res[kvalue][metric]['meanrank'] / len(allres), 2)

    return res

def trans(train_set):
    simlist = {}

    pca = PCA(n_components=train_set.shape[1])
    pca_train_set = pd.DataFrame(pca.fit_transform(train_set.T), index=train_set.T.index)
    simlist['PCA'] = similarity(pca_train_set.T)

    mca = prince.MCA(n_components=train_set.shape[1])
    mca_train_set = mca.fit_transform(train_set.T)
    simlist['MCA'] = similarity(pd.DataFrame(mca_train_set.T))

    tsne = manifold.TSNE(n_components=2048, method='exact')
    tsne_train_set = pd.DataFrame(tsne.fit_transform(train_set.T), index=train_set.T.index)
    simlist['TSNE'] = similarity(tsne_train_set.T)

    return simlist

if __name__ == '__main__':
    file = 'preprocess_recipes.csv'
    allres = []

    # 5-fold evaluation
    for state in range(60, 110, 10):
        train_set, test_set = splitData(file, state)
        simlist = trans(train_set)
        allres.append(completeRecipe(test_set, simlist, state))

    res = avgRes(allres)
    with open('reduction_result.json', 'w') as f:
        json.dump(res, f, indent=2)
