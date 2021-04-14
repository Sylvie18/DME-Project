# DME-Project

All the codes are written in python, setup a python3 virtual env and install the following packages:    
numpy, sklearn, pandas, scipy, matplotlib  
The data preprocessing part is defined in preprocessing.ipynb.  
The visualisation of recipe and generation of low dimension ingredients is written in Dimensionality_reduction.ipynb.  
All the exploratory data analysis processes are included in eda.ipynb.

---
## Model-based Collaborative Filtering
The model-based collaborative filtering in this project is based on the logistic regression. The code is in `model_CF.py`. The experiment result is shown in `result_test_LogisticRegression.json`. In the exeriment part, we test optimization method of NEWTON-CG, LBFGS, SAG, LIBLINEAR and SAGA with L1 or L2 regularization. The evaluation matrics is *Precision@10* and *mean rank*.

The experiment result is shown in table below.

### Precision@10
|Optimization | newton-cg | lbfgs | sag | liblinear | saga|
|  ----  | ----  |  ----  | ----  |  ----  | ----  |
|L1 regularization      | -  | -  | -  | 60.142   | 60.284  |
|L2 regularization    | 60.967  | 60.849  | **60.967**  | 60.731   | 60.967|
|


### mean rank
|Optimization | newton-cg | lbfgs | sag | liblinear | saga|
|  ----  | ----  |  ----  | ----  |  ----  | ----  |
|L1 regularization    | -  | -  | -  | 21.544  | 21.684|
|L2 regularization    | 21.022  | **21.011**  | 21.021  | 21.573   | 21.022|
| 

---