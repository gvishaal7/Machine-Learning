#the following application builds an artificial neural network using keras and tensorflow

from keras.models import Sequential
from keras.layers import Dense, Dropout
import numpy
import pandas as pd
from numpy import array
from sklearn.metrics import accuracy_score
from random import randrange
from random import random
# fix random seed for reproducibility
numpy.random.seed(7)

personal_data = #personal data 
predict_data = #readings from sensors (day to day data)
medical_data = #previous medical history

pd.set_option('display.float_format', lambda x: '%.0f' % x)

#data cleaning
medi_personal = pd.merge(personal_data,medical_data,on='ID')
personal_predic = pd.merge(medi_personal,predict_data,on='ID',how='right')
del personal_predic['name']
personal_predic['Time'] = pd.to_datetime(personal_predic['Time'],format='%m/%d/%y')
personal_predic['Time'] = personal_predic['Time'].apply(lambda x:x.timestamp()*1000)
personal_predic = personal_predic.sort_values('Time')
del personal_predic['ID']
del personal_predic['Time']

#data reshaping
dt = list()
dt = personal_predic.values.tolist()

#characteristic of the neural network
#8 hidden layers each with 8 nodes
#loss function = binary_crossentropy
#optimizer = adam 
#learning rate 0.01
model = Sequential()
model.add(Dense(8, input_dim=8, activation='relu')) #input layer
model.add(Dropout(0.25))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(8, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid')) #output layer
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

#funtion which splits dataset into train and test over the given fold value
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

#builds the neural network
def execute_model_ann(dataset,folds,e):
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
        model.fit(X, Y, epochs=e, batch_size=10, verbose=0)
        predictions = model.predict(test_array[:,0:8])
        rounded = [round(x[0]) for x in predictions]
        test_array_y = test_array[:,8]
        test_round = [round(x) for x in test_array_y]
        acc = accuracy_score(rounded, test_round, normalize=True)*100
        scores.append(acc)
    return scores

#testing the neural net's behavior for different epochs
epochs = [1700,3400,3000,4500,5000,7500]
for e in epochs:
    ss = execute_model_ann(dt,6,e)
    acc = sum(ss)/len(ss)
    print('Accuracy for epoch ',e,' is : ',acc)
