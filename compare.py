from numpy import genfromtxt
import numpy as np
from numpy import *
import matplotlib
import sklearn
#matplotlib.use('TKAgg') # matplotlib renderer for windows
#import TensorFlow_DNN_Heart_Disease as nn

import matplotlib.pyplot as plt
import TensorFlow_DNN_Heart_Disease as nn



from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
import pylab as pl
from itertools import cycle
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
from urllib.request import urlopen
#import urllib
#import matplotlib.pyplot as plt # Visuals
#import seaborn as sns 
import sklearn as skl
import pandas as pd
from sklearn import decomposition 
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

names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'heartdisease']
ClevelandHeartDisease = pd.read_csv('C:\\Users\\mural\\OneDrive\\Desktop\\AI\\Heart-Disease-Machine-Learning-master\\pdata.csv', names = names) #gets Cleveland data
Hungarian_data_URL = 'http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.hungarian.data'
Switzerland_data_URL = 'http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.switzerland.data'
datatemp = [ClevelandHeartDisease] #combines all arrays into a list
HungarianHeartDisease = pd.read_csv(urlopen(Hungarian_data_URL), names = names) #gets Hungary data
SwitzerlandHeartDisease = pd.read_csv(urlopen(Switzerland_data_URL), names = names) #gets Switzerland data
datatemp = [ClevelandHeartDisease, HungarianHeartDisease, SwitzerlandHeartDisease]
heartDisease = pd.concat(datatemp)#combines list into one array
heartDisease.head()
print(len(heartDisease))

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

def plot_2D(data,target,target_names):
	colors = cycle('rgbcmykw')
	target_ids = range(len(target_names))
	plt.figure()
	for i,c, label in zip(target_ids, colors, target_names):
		plt.scatter(data[target == i, 0], data[target == i, 1], c=c, label=label)
	plt.legend()
	plt.savefig('Problem 2 Graph')


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


def runsvm(train,test):
    import time
    start = time.clock()
    svmtest = svm.SVC(C=0.1,kernel='rbf')
    svmfit = svmtest.fit(train[['age', 'sex', 'cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak']], train['heartdisease'])
    tottrain = time.clock() - start
    print("\nTime taken to train SVM model is {0}".format(tottrain))
    
    start = time.clock()
    svmPredictions = svmtest.predict(test[['age', 'sex', 'cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak']])
    totpredict = time.clock() - start
    
    print("\nTime taken for prediction is {0}".format(totpredict))
    
    predictrightsvm = 0
    actlist =  []
    for i in range(0,svmPredictions.shape[0]-1):
        actlist.append(test.iloc[i][10])
        if (svmPredictions[i]== test.iloc[i][10]):
            predictrightsvm +=1
#    print("\nActual   Predicted\n")
#    for x,y in zip(actlist,svmPredictions):
#        print("{0}\t\t{1}".format(x,y))
    rightpercentsvm = predictrightsvm/svmPredictions.shape[0]
    print("\nAccuracy of the system using SVM is {0}".format(rightpercentsvm))
    return rightpercentsvm,tottrain,totpredict

# Decision Tree
def build_tree(train, test):
        
    import time
    start = time.clock()
    dt = tree.DecisionTreeClassifier(criterion= 'entropy')
    dt = dt.fit(train[['age', 'sex', 'cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak']], train['heartdisease'])
    tot1 = time.clock() - start
    print("\nTime taken to train Decision tree is {0}".format(tot1))
    import time
    start = time.clock()
    predictions_dt = dt.predict(test[['age', 'sex', 'cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak']])
    tot2 = time.clock() - start
    print("\nTime taken for prediction is {0}".format(tot2))
    
    predictright = 0
    actlistdt = []
    predictions_dt.shape[0]
    for i in range(0,predictions_dt.shape[0]-1):
        actlistdt.append(test.iloc[i][10])
        if (predictions_dt[i]== test.iloc[i][10]):
            predictright +=1
    #print("\nActual   Predicted\n")
    #for x,y in zip(actlistdt,predictions_dt):
    #    print("{0}\t\t{1}".format(x,y))
    accuracy = predictright/predictions_dt.shape[0]
    print("\nAccuracy of the system using Decision tree is {0}".format(accuracy))
    return accuracy, tot1, tot2



#SVM

mysplits=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
acclist = []
testsize = []
traint = []
predt =[]
for mysplit in mysplits:
    train, test = train_test_split(heartDisease, test_size = mysplit, random_state = 42)
    acc,traintime,predtime= runsvm(train,test)
    acclist.append(acc)
    testsize.append(mysplit)
    traint.append(traintime)
    predt.append(predtime)
print("\nTest size    \t  Accuracy \t\t\t     TrainTime \t\t \t    Predicttime\n")    
for w,x,y,z in zip(testsize,acclist,traint,predt):
    print("{0}\t\t{1}\t\t{2}\t\t{3}\t\t".format(w,x,y,z))




#Decision Tree
    
acclist_tree = []
testsize_tree = []
traint_tree = []
predt_tree =[]
    
for mysplit in mysplits:
    train, test = train_test_split(heartDisease, test_size = mysplit, random_state = 42)
    acc,traintime,predtime= build_tree(train,test)
    acclist_tree.append(acc)
    testsize_tree.append(mysplit)
    traint_tree.append(traintime)
    predt_tree.append(predtime)
print("\nTest size    \t  Accuracy \t\t\t     TrainTime \t\t \t    Predicttime\n")    
for w,x,y,z in zip(testsize_tree,acclist_tree,traint_tree,predt_tree):
    print("{0}\t\t{1}\t\t{2}\t\t{3}\t\t".format(w,x,y,z))
    

#Neural Network80.20210.

acclist_net = []
testsize_net = []
traint_net = []
predt_net =[]

for mysplit in mysplits:
    acc,traintime,predtime= nn.neural_network(mysplit)
    acclist_net.append(acc)
    testsize_net.append(mysplit)
    traint_net.append(traintime)
    predt_net.append(predtime)
print("\nTest size    \t  Accuracy \t\t\t     TrainTime \t\t \t    Predicttime\n")    
for w,x,y,z in zip(testsize_net,acclist_net,traint_net,predt_net):
    print("{0}\t\t{1}\t\t{2}\t\t{3}\t\t".format(w,x,y,z))