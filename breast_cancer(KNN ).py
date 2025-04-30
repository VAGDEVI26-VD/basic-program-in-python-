import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv(r"C:\Users\Lenovo\Documents\Notes\APRIL\11th - KNN\projects\KNN\brest cancer.txt")
data.head()

data.shape

data.describe()
data.isnull().sum()

col_names = ['Id', 'Clump_thickness', 'Uniformity_Cell_Size', 'Uniformity_Cell_Shape', 'Marginal_Adhesion', 'Single_Epithelial_Cell_Size', 'Bare_Nuclei', 'Bland_Chromatin', 'Normal_Nucleoli', 'Mitoses', 'Class']
data.columns = col_names
data.columns

data.head()

data.drop(['Id'],axis=1)
data.head()

data.info()

for var in data.columns:
    
    print(data[var].value_counts())
    
data['Bare_Nuclei'] = pd.to_numeric(data['Bare_Nuclei'], errors='coerce')
data.dtypes

data.isnull().sum()
data.isna().sum()

data['Bare_Nuclei'].value_counts()
data['Bare_Nuclei'].unique()
data['Bare_Nuclei'].isna().sum()

data['Class'].value_counts()
data['Class'].value_counts()/np.float64(len(data))

print(round(data.describe(),2))



#visualizing:
    
  

plt.rcParams['figure.figsize']=(30,25)

data.plot(kind='hist', bins=10, subplots=True, layout=(6, 2) ,sharex=False, sharey=False)

plt.show()


correlation = data.corr()
correlation['Class'].sort_values(ascending=False)


plt.figure(figsize=(10,8))
plt.title('Correlation of Attributes with Class variable')
a = sns.heatmap(correlation, square=True, annot=True, fmt='.2f', linecolor='white')
a.set_xticklabels(a.get_xticklabels(), rotation=90)
a.set_yticklabels(a.get_yticklabels(), rotation=30)           
plt.show()


x = data.drop(['Class'], axis=1)

y = data['Class']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

x_train.shape
x_test.shape
x_train.dtypes
x_train.isnull().sum()
x_test.isnull().sum()


for col in x_train.columns:
    if x_train[col].isnull().mean()>0:
        print(col, round(x_train[col].isnull().mean(),4))

for df1 in [x_train, x_test]:
    for col in x_train.columns:
        col_median=x_train[col].median()
        df1[col].fillna(col_median, inplace=True)    

x_train.isnull().sum()
x_test.isnull().sum()

x_train.head()
x_test.head()

cols = x_train.columns


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train=pd.DataFrame(x_train,columns=[cols])
x_test=pd.DataFrame(x_test,columns=[cols])

x_train.head()

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)

y_pred=knn.predict(x_test)
y_pred


knn.predict_proba(x_test)[:,0]
knn.predict_proba(x_test)[:,1]


from sklearn.metrics import accuracy_score

print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))

y_pred_train=knn.predict(x_train)
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))


print('Training set score: {:.4f}'.format(knn.score(x_train, y_train)))

print('Test set score: {:.4f}'.format(knn.score(x_test, y_test)))


y_test.value_counts()

null_accuracy = (85/(85+55))

print('Null accuracy score: {0:0.4f}'. format(null_accuracy))

#rebuild KNN classification:
     #with diff k-Values:
         
knn_5 = KNeighborsClassifier(n_neighbors=5)
knn_5.fit(x_train, y_train)
y_pred_5 = knn_5.predict(x_test)

print('Model accuracy score with k=5 : {0:0.4f}'. format(accuracy_score(y_test, y_pred_5)))


from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print('Confusion matrix\n\n', cm)

print('\nTrue Positives(TP) = ', cm[0,0])

print('\nTrue Negatives(TN) = ', cm[1,1])

print('\nFalse Positives(FP) = ', cm[0,1])

print('\nFalse Negatives(FN) = ', cm[1,0])























