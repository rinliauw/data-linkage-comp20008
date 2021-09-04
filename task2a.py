import pandas as pd
import numpy as np
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier

#Load in the data
life=pd.read_csv('life.csv',encoding = 'ISO-8859-1')
world=pd.read_csv('world.csv',encoding = 'ISO-8859-1')

#Dropping the rows that are not relevant to life.csv
life_countrycode = life['Country Code'].tolist()
unnecessary_countrycode = []

index = 0
for code in world['Country Code']:
    if code not in life_countrycode:
        unnecessary_countrycode.append(index)
    index += 1
world_data = world.drop(unnecessary_countrycode)
world_data.index = range(len(world_data))

#Replacing '..' with NaN
heading_name = world_data.columns.values.tolist()
for i in range(len(world_data['Country Code'])):
    for j in range(len(heading_name)):
        if world_data.iloc[i][j]=='..':
            world_data.iloc[i][j] = np.nan

life.index = life['Country Code']
#Make a new classlabel that is associated with the new rows in world_data
classlabel=[]
for code in world_data['Country Code']:
    classlabel.append(life.loc[code, 'Life expectancy at birth (years)'])

numeric_heading = heading_name[3:]
#Randomly select 2/3% of the instances to be training and the rest to be testing. split data and classlabel such that training size is 2/3
X_train, X_test, y_train, y_test = train_test_split(world_data[numeric_heading].astype(float),classlabel, train_size=2/3, test_size=1/3, random_state=100)

rounded_median_list = []
rounded_mean_list = []
rounded_variance_list = []

#Computes current median, imputes missing data with current median, then calculates mean and variance
for name in numeric_heading:
    current_median = X_train[name].median()
    #Imputes NaN with current median
    X_train[name].fillna(current_median, inplace = True)
    X_test[name].fillna(current_median, inplace = True)
    current_mean = X_train[name].mean()
    current_variance = X_train[name].var()
    rounded_median_list.append(round(current_median,3))
    rounded_mean_list.append(round(current_mean,3))
    rounded_variance_list.append(round(current_variance,3))

#Exports to csv
task2a_dict = {'feature': numeric_heading, 'median': rounded_median_list, 'mean': rounded_mean_list, 'variance': rounded_variance_list}
task2a_df = pd.DataFrame(task2a_dict)
task2a_df.to_csv("task2a.csv", index=False)

#Standarizes each X_train and X_test, by fitting X_train
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test=scaler.transform(X_test)

#Fits Decision Tree, K-Neighbors Classifier with K= 5,10 and prints accuracy 
dt = DecisionTreeClassifier(criterion="entropy",random_state=1, max_depth=4)
dt.fit(X_train, y_train)
y_pred=dt.predict(X_test)
print('Accuracy of decision tree:' ,round(accuracy_score(y_test, y_pred),3))

knn5 = neighbors.KNeighborsClassifier(n_neighbors=5)
knn5.fit(X_train, y_train)
y_pred=knn5.predict(X_test)
print('Accuracy of k-nn (k=5):',round(accuracy_score(y_test, y_pred),3))

knn10 = neighbors.KNeighborsClassifier(n_neighbors=10)
knn10.fit(X_train, y_train)
y_pred=knn10.predict(X_test)
print('Accuracy of k-nn (k=10):',round(accuracy_score(y_test, y_pred),3))