import numpy as np
from urllib.request import urlopen
#import urllib
#import matplotlib.pyplot as plt # Visuals
#import seaborn as sns 
import sklearn as skl
import pandas as pd
import TensorFlow_DNN_Heart_Disease as nn

from sklearn.model_selection import train_test_split # Create training and test sets
from sklearn.neighbors import KNeighborsClassifier # Kth Nearest Neighbor
from sklearn.tree import DecisionTreeClassifier # Decision Trees
from sklearn.tree import export_graphviz # Extract Decision Tree visual
from sklearn.tree import tree 
from sklearn.ensemble import RandomForestClassifier # Random Forest
from sklearn import svm #SVM
from sklearn.metrics import roc_curve # ROC Curves
from sklearn.metrics import auc # AUC 
from sklearn.model_selection import KFold, cross_val_score #cross validation 
from sklearn import model_selection  #cross validation 
from urllib.request import urlopen # Get data from UCI Machine Learning Repository


np.set_printoptions(threshold=np.nan) #see a whole array when we output it
split_size = [0.1,0.2,0.3,0.4,0.5]

names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'heartdisease']
ClevelandHeartDisease = pd.read_csv('C:\\Users\\mural\\OneDrive\\Desktop\\AI\\Heart-Disease-Machine-Learning-master\\pdata.csv', names = names) #gets Cleveland data
datatemp = [ClevelandHeartDisease] #combines all arrays into a list

heartDisease = pd.concat(datatemp)#combines list into one array
heartDisease.head()

del heartDisease['ca']
del heartDisease['slope']
del heartDisease['thal']

heartDisease = heartDisease.replace('?', np.nan)
heartDisease.dtypes

count = 0
for item in heartDisease:
    for i in heartDisease[item]:
        count += (i == '?')



for item in heartDisease: #converts everything to floats
    heartDisease[item] = pd.to_numeric(heartDisease[item])

def normalize(heartDisease, toNormalize): #normalizes 
    result = heartDisease.copy()
    for item in heartDisease.columns:
        if (item in toNormalize):
            max_value = heartDisease[item].max()
            min_value = heartDisease[item].min()
            result[item] = (heartDisease[item] - min_value) / (max_value - min_value)
    return result
toNormalize = ['age', 'cp', 'trestbps', 'chol', 'thalach', 'oldpeak'] #columns to normalize
heartDisease = normalize(heartDisease, toNormalize)
heartDisease = heartDisease.dropna()
heartDisease.head()

for i in range(1,5):
    heartDisease['heartdisease'] = heartDisease['heartdisease'].replace(i,1)


train, test = train_test_split(heartDisease, test_size = 0.1, random_state = 42)

#SVM
import time
start = time.clock()
svmtest = svm.SVC(C=0.1,kernel='rbf')
svmfit = svmtest.fit(train[['age', 'sex', 'cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak']], train['heartdisease'])
tot = time.clock() - start
print("\nTime taken to train SVM model is {0}".format(tot))

start = time.clock()
svmPredictions = svmtest.predict(test[['age', 'sex', 'cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak']])
tot = time.clock() - start

print("\nTime taken for prediction is {0}".format(tot))

predictrightsvm = 0
actlist =  []
for i in range(0,svmPredictions.shape[0]-1):
    actlist.append(test.iloc[i][10])
    if (svmPredictions[i]== test.iloc[i][10]):
        predictrightsvm +=1
print("\nActual   Predicted\n")
for x,y in zip(actlist,svmPredictions):
    print("{0}\t\t{1}".format(x,y))
rightpercentsvm = predictrightsvm/svmPredictions.shape[0]
print("\nAccuracy of the system using SVM is {0}".format(rightpercentsvm))


# Decision Tree
import time
start = time.clock()
dt = tree.DecisionTreeClassifier(criterion= 'entropy')
dt = dt.fit(train[['age', 'sex', 'cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak']], train['heartdisease'])
tot = time.clock() - start
print("\nTime taken to train Decision tree is {0}".format(tot))
import time
start = time.clock()
predictions_dt = dt.predict(test[['age', 'sex', 'cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak']])
tot = time.clock() - start
print("\nTime taken for prediction is {0}".format(tot))
print(dt.max_depth)

predictright = 0
actlistdt = []
predictions_dt.shape[0]
for i in range(0,predictions_dt.shape[0]-1):
    actlistdt.append(test.iloc[i][10])
    if (predictions_dt[i]== test.iloc[i][10]):
        predictright +=1
print("\nActual   Predicted\n")
for x,y in zip(actlistdt,predictions_dt):
    print("{0}\t\t{1}".format(x,y))
accuracy = predictright/predictions_dt.shape[0]
print("\nAccuracy of the system using Decision tree is {0}".format(accuracy))


for split in split_size:
    nn.neural_network(split)