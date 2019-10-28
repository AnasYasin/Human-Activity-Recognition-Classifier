import sklearn
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectPercentile

df_train_X = pd.read_csv('final_X_train.txt')
df_train_y = pd.read_csv('final_y_train.txt')
df_test_X = pd.read_csv('final_X_test.txt')
df_test_y = pd.read_csv('final_y_test.txt')

df_train_X.dropna(0, inplace=True)
df_test_X.dropna(0, inplace=True)


'''
#pca
pca = PCA(n_components=100)
df_train_X = pca.fit_transform(df_train_X)
df_test_X = pca.fit_transform(df_test_X)
'''

#feature selection varience 
select = SelectPercentile(percentile=90)
df_train_X = select.fit_transform(df_train_X,df_train_y)
df_test_X = select.transform(df_test_X)


#feature scaling
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
df_train = min_max_scaler.fit_transform(df_train_X)
df_test = min_max_scaler.fit_transform(df_test_X)

#training 
svm=SVC(gamma=0.1, kernel='rbf', C=3)
svm.fit(df_train, df_train_y)   

#prediction
y_train_predicted=svm.predict(df_train)
y_test_predicted=svm.predict(df_test)

#Accuracy
print('F1 Score of Each Class:', sklearn.metrics.f1_score(df_test_y, y_test_predicted, average=None, labels=None, pos_label=1,  sample_weight=None))
print('Train accuracy:', sklearn.metrics.f1_score(df_train_y, y_train_predicted, labels=None, pos_label=1, average="micro", sample_weight=None))
print('Test accuracy:', sklearn.metrics.f1_score(df_test_y.astype(int), y_test_predicted.astype(int), labels=None, pos_label=1, average="micro", sample_weight=None))

#np.savetxt('predictions.txt', y_test_predicted)
