import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file = (r"C:\Users\Lenovo\Documents\Notes\APRIL\11th - KNN\projects\LOGISTIC REGRESSION , PCA, EDA\adult\adult.csv")
df = pd.read_csv(file, encoding='latin-1')
df

df.shape
df.head()
df.info()

df[df=='?']=np.nan
df.head()

df.info()

df.columns

for col in ['workclass', 'occupation', 'native.country']:df[col].fillna(df[col].mode()[0],inplace=True)

df.isnull().sum()

x=df.drop(['income'],axis=1)
y=df['income']

x.head()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

x=df.drop(['income'],axis=1)
y=df['income']

from sklearn import preprocessing
categorical = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']
for feature in categorical:
        le = preprocessing.LabelEncoder()
        x_train[feature] = le.fit_transform(x_train[feature])
        x_test[feature] = le.transform(x_test[feature])
        
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train=pd.DataFrame(scaler.fit_transform(x_train),columns=x.columns)
x_test=pd.DataFrame(scaler.transform(x_test),columns=x.columns)

x_train.head()

#logistic reg:

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

logreg=LogisticRegression()
logreg.fit(x_train,y_train)
y_pred=logreg.predict(x_test)

print('Logistic Regression accuracy score with all the features: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


#PCA with logistic regression:
    
from sklearn.decomposition import PCA
pc=PCA()
x_train=pc.fit_transform(x_train)
pc.explained_variance_ratio_


#log reg with 13 features:
    
x=df.drop(['income','native.country'],axis=1)
y=df['income']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

categorical = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex']
for feature in categorical:
    le = preprocessing.LabelEncoder()
    x_train[feature] = le.fit_transform(x_train[feature])
    x_test[feature] = le.transform(x_test[feature])

x_train=pd.DataFrame(scaler.fit_transform(x_train),columns=x.columns)
x_test=pd.DataFrame(scaler.transform(x_test),columns=x.columns)


logreg=LogisticRegression()
logreg.fit(x_train,y_train)
y_pred=logreg.predict(x_test)


print('Logistic Regression accuracy score with the 13 features: {0:0.4f}'. format(accuracy_score(y_test, y_pred)))


#log reg with 12 features:
    
x=df.drop(['income','native.country','hours.per.week'],axis=1)
y=df['income']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

categorical= ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex']
for feature in categorical:
    le=preprocessing.LabelEncoder()
    x_train[feature]=le.fit_transform(x_train[feature])
    x_test[feature]=le.transform(x_test[feature])

x_train=pd.DataFrame(scaler.fit_transform(x_train),columns=x.columns)

x_test=pd.DataFrame(scaler.transform(x_test),columns=x.columns)

logreg=LogisticRegression()
logreg.fit(x_train,y_train)
y_pred=logreg.predict(x_test)

print('Logistic regression acurracy score with the 1st 12 feature:{0:0.4f}'. format(accuracy_score(y_test, y_pred)))

# 11 features

x=df.drop(['income','native.country','hours.per.week','capital.loss'],axis=1)

y=df['income']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

categorical=['workclass','education','marital.status','occupation','relationship','race','sex']
for feature in categorical:
    le=preprocessing.LabelEncoder()
    x_train[feature]=le.fit_transform(x_train[feature])
    x_test[feature]=le.transform(x_test[feature])
    
x_train=pd.DataFrame(scaler.fit_transform(x_train),columns=x.columns)
x_test=pd.DataFrame(scaler.transform(x_test),columns=x.columns)

logreg=LogisticRegression()
logreg.fit(x_train,y_train)
y_pred=logreg.predict(x_test)


print('Logistic regression acurracy score with the 1st 11 feature:{0:0.4f}'. format(accuracy_score(y_test, y_pred)))


#select right number of dimensions:
    

x=df.drop(['income'],axis=1)
y=df['income']


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)


le_dict = {}

categorical = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'sex']


for feature in categorical:
    le=preprocessing.LabelEncoder()
    x_train[feature] = le.fit_transform(x_train[feature])
    le_dict[feature] = le


x_train=pd.DataFrame(scaler.fit_transform(x_train),columns=x.columns)



pca= PCA()
pca.fit(x_train)
cumsum = np.cumsum(pca.explained_variance_ratio_)
dim = np.argmax(cumsum >= 0.90) + 1
print('The number of dimensions required to preserve 90% of variance is',dim)

#visualization:
    
plt.figure(figsize=(8,6))
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlim(0,14,1)
plt.xlabel('Number of components')
plt.ylabel('Cumulative explained variance')
plt.show()













