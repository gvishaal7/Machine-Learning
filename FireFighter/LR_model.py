#the following application builds a linear model over the given dataset

import pandas as pd
import time
from numpy import array
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from random import randrange

personal_data = #personal data 
medical_data = #readings from sensors (day to day data)
predict_data = #previous medical history

pd.set_option('display.float_format', lambda x: '%.0f' % x)

#data cleaning
personal_predic = pd.merge(personal_data,predict_data,on='ID',how='right')
medi_personal = pd.merge(personal_data,medical_data,on='ID')
personal_predic = pd.merge(medi_personal,predict_data,on='ID',how='right')
del personal_predic['name']
personal_predic['Time'] = pd.to_datetime(personal_predic['Time'],format='%m/%d/%y')
personal_predic['Time'] = personal_predic['Time'].apply(lambda x:x.timestamp()*1000)
personal_predic = personal_predic.sort_values(['Time','age'])
del personal_predic['ID']
del personal_predic['Time']

#data reshaping
dt = list()
dt = personal_predic.values.tolist()

#funtion which splits dataset into train and test over the given fold values
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

#builds the linear regression model
def execute_model(dataset,folds):
    folds = cross_validation_split(dataset,folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] =None
        train_array = array(train_set)
        test_array = array(test_set)
        X = train_array[:,0:8]
        Y = train_array[:,8]
        x_test = test_array[:,0:8]
        y_test = test_array[:,8]
        lm = linear_model.LinearRegression()
        model = lm.fit(X,Y)
        t = lm.predict(x_test)
        t_round = [round(x) for x in t]
        y_test_round = [round(x) for x in y_test]
        acc = accuracy_score(t_round, y_test_round, normalize=True)*100
        scores.append(acc)
    return scores

tot_acc = 0
folds = [2,3,4,6,12,17]
for fold in folds:
    ss = execute_model(dt,fold)
    acc = sum(ss)/len(ss)
    tot_acc = tot_acc + acc
    print('Accuracy for fold',fold,' is : 'acc)

print('Average accuracy over the folds is',(tot_acc/6))
