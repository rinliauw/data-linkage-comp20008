import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import neighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, pairwise_distances_argmin_min
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif

###Copied code from task 2a

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

#Imputes X_train and X_test by median value of X_train
for name in heading_name[3:]:
    current_median = X_train[name].median()
    X_train[name].fillna(current_median, inplace = True)
    X_test[name].fillna(current_median, inplace = True)

###End of copied code from task 2a

features = world_data[heading_name[3:]].astype(float)

#Copies to update X_train
X_train_copy_2 = X_train.copy()
X_test_copy_2 = X_test.copy()
X_train_copy_2.columns = heading_name[3:]
X_test_copy_2.columns = heading_name[3:]

#Gets all interaction term pairs and updates X_train_copy_2 and X_test_copy_2
i=3
for f1 in heading_name[3:len(heading_name)-1]:
    for f2 in heading_name[i+1:]:
        X_train_copy_2['{}*{}'.format(f1,f2)] = X_train_copy_2[f1] * X_train_copy_2[f2]
        X_test_copy_2['{}*{}'.format(f1,f2)] = X_test_copy_2[f1] * X_test_copy_2[f2]
    i+=1

#Evidence of interaction term pairs
print("#X_train's column after Interaction Term Pairs:",len(X_train_copy_2.columns))
print("#X_test's column after Interaction Term Pairs:",len(X_test_copy_2.columns))
print('X_train of 211 features: \n', X_train_copy_2, 'X_test of 211 features: \n', X_test_copy_2)

#Generate K-Means
points = np.array(X_train[heading_name[3:]]) #data points

#Create elbow plot: code segment from https://blog.cambridgespark.com/how-to-determine-the-optimal-number-of-clusters-for-k-means-clustering-14f27070048f
sum_of_squared_distances = []
K = range(1,6)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(points)
    sum_of_squared_distances.append(km.inertia_)

#Evidence for choosing K-Means where K = 3
plt.figure()
plt.annotate('Elbow',(3, 1e9))
plt.plot(K, sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method for Optimal k')
print('Clustering: using elbow method, k value is 3, so we choose k = 3')
#Save figure
plt.savefig("task2bgraph1.png", bbox_inches="tight")

clusters = KMeans(n_clusters=3).fit(points)
print('X_train cluster labels:', clusters.labels_)

X_train_copy_2['fclusterlabel'] = clusters.labels_
x,y = pairwise_distances_argmin_min(X_test, clusters.cluster_centers_) #calculate closest distance
X_test_copy_2['fclusterlabel'] = x
print('X_test cluster labels assigned to nearest centroids:', X_test_copy_2['fclusterlabel'].tolist())

#Selecting 4 features. use mutual information

X = X_train_copy_2 #independent columns
y = y_train #target column i.e price range
#Apply SelectKBest class to extract top 10 best features, code segment from:
#https://www.google.com/search?sxsrf=ALeKk03g1zLuwQsIFAeL-6o7fCAtu5vK4Q%3A1590856369850&ei=sYrSXp6-MrPVz7sPicaywAo&q=fit+%3D+bestfeatures.fit%28X%2Cy%29+dfscores+%3D+pd.DataFrame%28fit.scores_%29+dfcolumns+%3D+pd.DataFrame%28X.columns%29+featureScores+%3D+pd.concat%28%5Bdfcolumns%2Cdfscores%5D%2Caxis%3D1%29+featureScores.columns+%3D+%5B%27Specs%27%2C%27Score%27%5D+&oq=fit+%3D+bestfeatures.fit%28X%2Cy%29+dfscores+%3D+pd.DataFrame%28fit.scores_%29+dfcolumns+%3D+pd.DataFrame%28X.columns%29+featureScores+%3D+pd.concat%28%5Bdfcolumns%2Cdfscores%5D%2Caxis%3D1%29+featureScores.columns+%3D+%5B%27Specs%27%2C%27Score%27%5D+&gs_lcp=CgZwc3ktYWIQAzIECAAQRzIECAAQRzIECAAQRzIECAAQRzIECAAQRzIECAAQRzIECAAQRzIECAAQR1CoBFioBGDKBWgAcAN4AIABAIgBAJIBAJgBAKABAqABAaoBB2d3cy13aXo&sclient=psy-ab&ved=0ahUKEwje2r_mgdzpAhWz6nMBHQmjDKgQ4dUDCAw&uact=5
bestfeatures = SelectKBest(score_func=mutual_info_classif, k='all')
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Heading','Score']
featureScores = featureScores.sort_values(by = 'Score', ascending = False)

top_4_features = featureScores[:4]
print('Top 4 Scores According to MI by SelectKBest:\n',featureScores.nlargest(4,'Score'))

#Make a DataFrame of the top 4 headings with their data values
mii_index = top_4_features['Heading']
X_train_selection = {}
X_test_selection = {}
for heading_top4 in mii_index:
    X_train_selection[heading_top4] = X_train_copy_2[heading_top4]
    X_test_selection[heading_top4] = X_test_copy_2[heading_top4]
X_train_selection = pd.DataFrame(X_train_selection)
X_test_selection = pd.DataFrame(X_test_selection)

#Apply Standarization before K-NN: need to be normalized since vectors
scaler = preprocessing.StandardScaler().fit(X_train_selection)
X_train_selection = scaler.transform(X_train_selection)
X_test_selection=scaler.transform(X_test_selection)

#Applying K-NN
knn_selection = neighbors.KNeighborsClassifier(n_neighbors=5)
knn_selection.fit(X_train_selection, y_train)
y_pred_selection=knn_selection.predict(X_test_selection)

#Below are code for PCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#Apply Standarization: need to be normalized since vectors
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test=scaler.transform(X_test)

import seaborn as sns
#Applying PCA: extracted code segment from https://www.geeksforgeeks.org/principal-component-analysis-with-python/
pca = PCA(n_components = 4)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_
print('Explained variance: ', explained_variance)

#For evidence of why PCA and selecting 4 best isn't really good
#https://www.kindsonthegenius.com/2019/01/12/principal-components-analysispca-in-python-step-by-step/
percent_variance = np.round(pca.explained_variance_ratio_* 100, decimals =2)
columns = ['PC1', 'PC2', 'PC3', 'PC4']
plt.figure()
plt.bar(x= range(1,5), height=percent_variance, tick_label=columns)
plt.ylabel('Percentate of Variance Explained')
plt.xlabel('Principal Component')
plt.title('PCA Scree Plot')
plt.savefig("task2bgraph2.png", bbox_inches="tight")

knn_pca = neighbors.KNeighborsClassifier(n_neighbors=5)
knn_pca.fit(X_train_pca, y_train)
y_pred_pca=knn_pca.predict(X_test_pca)

#Below are code for selecting first 4 Rows: D-G
X_train_4, X_test_4, y_train_4, y_test_4 = train_test_split(world_data[heading_name[3:7]].astype(float),classlabel, train_size=2/3, test_size=1/3, random_state=100)

for name in heading_name[3:7]:
    current_median = X_train_4[name].median()
    X_train_4[name].fillna(current_median, inplace = True)
    X_test_4[name].fillna(current_median, inplace = True)

#Apply Standarization
scaler = preprocessing.StandardScaler().fit(X_train_4)
X_train_4 = scaler.transform(X_train_4)
X_test_4=scaler.transform(X_test_4)

print_dataframe = pd.DataFrame(X_train_4)
print_dataframe.columns = heading_name[3:7]
print('Normalized X_train for headings in column D-G: \n', print_dataframe)
#Accuracy of first four features
knn_first4 = neighbors.KNeighborsClassifier(n_neighbors=5)
knn_first4.fit(X_train_4, y_train_4)
y_pred2=knn_first4.predict(X_test_4)

#Prints all accuracy value
print('Accuracy of feature engineering:','{:.3f}'.format(round(accuracy_score(y_test, y_pred_selection),3)))
print('Accuracy of PCA:','{:.3f}'.format(round(accuracy_score(y_test, y_pred_pca),3)))
print('Accuracy of first four features:','{:.3f}'.format(round(accuracy_score(y_test_4, y_pred2),3)))
