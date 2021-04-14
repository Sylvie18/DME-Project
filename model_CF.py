import pandas as pd
import numpy as np
from math import sqrt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import operator
import random
import json

# split dataset into train and test
def splitData(data):
    # data = pd.read_csv(file)
    train_set, test_set = train_test_split(data, test_size=0.2, random_state=18)

    return train_set, test_set

def convertDict(data):
    res = {}
    for i in range(len(data)):
        line = data.iloc[[i]]
        line = line.loc[:, (line != 0).any(axis=0)]
        res.update(line.to_dict(orient='index'))

    return res

def trainGenerator(trainData):
    print(trainData.shape)
    columnsName = trainData.columns.values.tolist()
    recipesDict = convertDict(trainData)
    allTrainData_x = []
    allTrainData_y = []
    for _, recipe in recipesDict.items():
        ingredientList = list(recipe.keys())
        for i in range(len(ingredientList)):
            tmpIngredientList = ingredientList.copy()
            missingIngredient = tmpIngredientList.pop(i)
            allTrainData_y.append(columnsName.index(missingIngredient))
            tmpList = []
            for oneIngredient in columnsName:
                if oneIngredient in tmpIngredientList:
                    tmpList.append(1)
                else:
                    tmpList.append(0)
            allTrainData_x.append(tmpList)
    x_train = pd.DataFrame(allTrainData_x, columns=columnsName)
    print(x_train.shape)
    y_train = pd.DataFrame(allTrainData_y)[0]
    print(y_train.shape)
    return x_train, y_train

def dictToFrame(cuisinesDict, columnsNameList):
    indexList = []
    dataList = []
    for i, recipe in cuisinesDict.items():
        indexList.append(i)
        ingredientList = list(recipe.keys())
        tmpList = []
        for oneIngredient in columnsNameList:
            if oneIngredient in ingredientList:
                tmpList.append(1)
            else:
                tmpList.append(0)
        dataList.append(tmpList)
    cuisineDataFrame = pd.DataFrame(dataList, index=indexList,columns=columnsNameList)
    return cuisineDataFrame

def make_missingIngs(data, state=0):
    misset = convertDict(data)
    y_test = []
    random.seed(state)
    for i, ingList in misset.items():
        misIng = random.choice(list(ingList))
        y_test.append(misIng)
        del (misset[i][misIng])
    x_test = dictToFrame(misset, data.columns.values.tolist())
    return x_test, y_test

def recommendListCalculate(probaFrame, y_testList):
    probDict = convertDict(probaFrame)
    rank = []
    precision10 = []

    for i, probas in probDict.items():
        count=0
        for ingredient, score in sorted(probas.items(), key=operator.itemgetter(1), reverse=True):
            if ingredient == y_testList[i]:
                rank.append(count)
                if count < 10:
                    precision10.append(1)
                else:
                    precision10.append(0)
                break
            else:
                count +=1
    precision = sum(precision10)/len(precision10)
    meanRank = sum(rank)/len(rank)
    print('precision: ', precision, '\tmeanrank: ', meanRank)

    return precision, meanRank

def logisticRegressionModel(trainDataFrame, testDataFrame, random_state=0):
    columnsName = testDataFrame.columns.values.tolist()
    # generate bigger train set
    print('Generate bigger train dataset')
    x_train, y_train = trainGenerator(train_set)
    print(y_train.unique())
    print(len(columnsName))
    # test set prepare
    print('test set prepare')
    x_test, y_testList = make_missingIngs(testDataFrame, random_state)
    print(x_test.shape)
    print(len(y_testList))

    lrSettingList = [['liblinear','l2'],['newton-cg','l2'],['lbfgs','l2'],['sag','l2'],['liblinear','l1'],['saga','l1'],['saga','l2']]
    resultDict = {}
    for setting in  lrSettingList:
        print('Setting: ',setting)
        # fitting
        print('fitting the Logistic Regression Model')
        if setting[0] == None or setting[1] == None:
            lr = LogisticRegression()
        else:
            lr = LogisticRegression(solver=setting[0],penalty=setting[1])
        lr.fit(x_train, y_train)
        # testing
        print('Testing the model')
        probaArray = lr.predict_proba(x_test)
        probaFrame = pd.DataFrame(probaArray, columns=columnsName[:probaArray.shape[1]])
        precision, meanRank = recommendListCalculate(probaFrame, y_testList)
        resultDict[str(setting)] = {'precision': precision,'meanrank': meanRank}
    return resultDict
    

if __name__ == '__main__':
    # file = 'modified_data-2.csv'
    file = 'preprocess_recipes.csv'
    data = pd.read_csv(file)

    train_set, test_set = splitData(data)
    randomState=0

    res = logisticRegressionModel(train_set, test_set, random_state=randomState)
    if res != None:
        with open('result_test_LogisticRegression.json', 'w') as f:
            json.dump(res, f, indent=2)