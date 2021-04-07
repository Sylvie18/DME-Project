import pandas as pd
from sklearn.model_selection import train_test_split
from math import sqrt


# construct the inverted index of user -> item
def convertDict(data):
    res = {}
    for i in range(len(data)):
        line = data.iloc[[i]]
        line = line.loc[:, (line != 0).any(axis=0)]
        res.update(line.to_dict(orient='index'))

    return res

# split dataset into train and test
def splitData(file):
    data = pd.read_csv(file)
    train_set, test_set = train_test_split(data, test_size=0.2, random_state=18)
    return convertDict(train_set), convertDict(test_set)

# calculate cosine similarity
def simCos(Mi, Mij):
    res = {}
    for i, itemlist in Mij.items():
        res.setdefault(i, {})
        for j in itemlist.keys():
            res[i].setdefault(j, 0)
            res[i][j] = Mij[i][j] / sqrt(Mi[i] * Mi[j])

    return res

# calculate co-occurrence matrix of item -> item
# calculate similarity matrix
def similarity(data):
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

    simlist = []
    simlist.append(simCos(Mi, Mij))

    return simlist


if __name__ == '__main__':
    file = 'recipes.csv'
    train_set, test_set = splitData(file)
    simlist = similarity(train_set)
    print(simlist)

