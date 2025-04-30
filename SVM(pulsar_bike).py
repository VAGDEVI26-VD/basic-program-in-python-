import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv(r"C:\Users\Lenovo\Documents\Notes\APRIL\10th - SVM\SVM\pulsar_data_test.csv")
df

df.shape
df.describe()
df.info()
df.head()
df.columns
#col_names=df.columns
#col_names

df.columns=df.columns.str.strip()
df.columns

df.columns= ['IP Mean', 'IP Sd', 'IP Kurtosis', 'IP Skewness','DM-SNR Mean', 'DM-SNR Sd', 'DM-SNR Kurtosis', 'DM-SNR Skewness', 'target_class']
df.columns

df['target_class'].value_counts()
print(df['target_class'].value_counts())

# Set 'target_class' missing values to zero
df['target_class'] = df['target_class'].fillna(0)

# Set missing values in all columns to zero
df = df.fillna(0)


df['target_class'].value_counts()/float(len(df))

df.info()
df.isnull().sum()

round(df.describe(),2)


#visuals: of outliers in boxplot

plt.figure(figsize=(24,20))


plt.subplot(4, 2, 1)
fig = df.boxplot(column='IP Mean')
fig.set_title('')
fig.set_ylabel('IP Mean')


plt.subplot(4, 2, 2)
fig = df.boxplot(column='IP Sd')
fig.set_title('')
fig.set_ylabel('IP Sd')


plt.subplot(4, 2, 3)
fig = df.boxplot(column='IP Kurtosis')
fig.set_title('')
fig.set_ylabel('IP Kurtosis')


plt.subplot(4, 2, 4)
fig = df.boxplot(column='IP Skewness')
fig.set_title('')
fig.set_ylabel('IP Skewness')


plt.subplot(4, 2, 5)
fig = df.boxplot(column='DM-SNR Mean')
fig.set_title('')
fig.set_ylabel('DM-SNR Mean')


plt.subplot(4, 2, 6)
fig = df.boxplot(column='DM-SNR Sd')
fig.set_title('')
fig.set_ylabel('DM-SNR Sd')


plt.subplot(4, 2, 7)
fig = df.boxplot(column='DM-SNR Kurtosis')
fig.set_title('')
fig.set_ylabel('DM-SNR Kurtosis')


plt.subplot(4, 2, 8)
fig = df.boxplot(column='DM-SNR Skewness')
fig.set_title('')
fig.set_ylabel('DM-SNR Skewness')

#histogram of distribution variables:
    

plt.figure(figsize=(24,20))


plt.subplot(4, 2, 1)
fig = df['IP Mean'].hist(bins=20)
fig.set_xlabel('IP Mean')
fig.set_ylabel('Number of pulsar stars')


plt.subplot(4, 2, 2)
fig = df['IP Sd'].hist(bins=20)
fig.set_xlabel('IP Sd')
fig.set_ylabel('Number of pulsar stars')


plt.subplot(4, 2, 3)
fig = df['IP Kurtosis'].hist(bins=20)
fig.set_xlabel('IP Kurtosis')
fig.set_ylabel('Number of pulsar stars')



plt.subplot(4, 2, 4)
fig = df['IP Skewness'].hist(bins=20)
fig.set_xlabel('IP Skewness')
fig.set_ylabel('Number of pulsar stars')



plt.subplot(4, 2, 5)
fig = df['DM-SNR Mean'].hist(bins=20)
fig.set_xlabel('DM-SNR Mean')
fig.set_ylabel('Number of pulsar stars')



plt.subplot(4, 2, 6)
fig = df['DM-SNR Sd'].hist(bins=20)
fig.set_xlabel('DM-SNR Sd')
fig.set_ylabel('Number of pulsar stars')



plt.subplot(4, 2, 7)
fig = df['DM-SNR Kurtosis'].hist(bins=20)
fig.set_xlabel('DM-SNR Kurtosis')
fig.set_ylabel('Number of pulsar stars')


plt.subplot(4, 2, 8)
fig = df['DM-SNR Skewness'].hist(bins=20)
fig.set_xlabel('DM-SNR Skewness')
fig.set_ylabel('Number of pulsar stars')


#decakre feature vector and target:
    
x=df.drop(['target_class'],axis=1)
y=df['target_class']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)


x_train.shape
x_test.shape

cols = x_train.columns

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

x_train = pd.DataFrame(x_train, columns=[cols])
x_test = pd.DataFrame(x_test, columns=[cols])

x_train.describe()










#############################???????????????

from sklearn.svm import SVC
lg_classifier =SVC() #kernal='poly',degree=3,linear==>for linear svm,rest off is non-linear svm
lg_classifier.fit(x_train,y_train)

y_pred=lg_classifier.predict(x_test)


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
cm

from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test, y_pred)
ac

bais=lg_classifier.score(x_train,y_train)
bais

variance=lg_classifier.score(x_test,y_test)
variance

from sklearn.metrics import classification_report
cr=classification_report(y_test, y_pred)
cr



