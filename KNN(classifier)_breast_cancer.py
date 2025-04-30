import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns

data = pd.read_csv(r"C:\Users\Lenovo\Documents\Notes\APRIL\22nd- Cross validation\breast-cancer-wisconsin.data.txt")
data.head()

data.shape


col_names = ['Id', 'Clump_thickness', 'Uniformity_Cell_Size', 'Uniformity_Cell_Shape', 'Marginal_Adhesion', 
             'Single_Epithelial_Cell_Size', 'Bare_Nuclei', 'Bland_Chromatin', 'Normal_Nucleoli', 'Mitoses', 'Class']

data.columns = col_names

data.columns
data.head()

data.drop('Id', axis=1,errors='ignore', inplace=True,)


data.info()


for var in data.columns:
    
    print(data[var].value_counts())
    

data['Bare_Nuclei'] = pd.to_numeric(data['Bare_Nuclei'], errors='coerce')

data.dtypes
data.isnull().sum()
data.isna().sum()
data['Bare_Nuclei'].value_counts()
     
data['Bare_Nuclei'].unique   
data['Bare_Nuclei'].isna().sum()     


data['Class'].value_counts()
data['Class'].value_counts()/float(len(data))

print(round(data.describe(),2))




#visualization:
    
plt.rcParams['figure.figsize']=(30,25)

data.plot(kind='hist', bins=10, subplots=True,layout=(5,2), sharex=False, sharey=False)

plt.show()


correlation=data.corr()
correlation['Class'].sort_values(ascending=False)


#heatmap:
    
plt.figure(figsize=(10,8))
plt.title('Correlation of Attributes with Class variable')
a = sns.heatmap(correlation, square=True, annot=True, fmt='.2f', linecolor='white')
a.set_xticklabels(a.get_xticklabels(), rotation=90)
a.set_yticklabels(a.get_yticklabels(), rotation=30)           
plt.show()


x=data.drop(['Class'],axis=1)
y=data['Class']

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


# scaling:
    

cols=x_train.columns

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)


x_train=pd.DataFrame(x_train,columns=[cols])
x_test=pd.DataFrame(x_test,columns=[cols])

x_train.head


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()
knn.fit(x_train, y_train)

y_pred=knn.predict(x_test)
y_pred


#probability of getting 2---->begin a cancer:
    
knn.predict_proba(x_test)[:,0]


# accuracy:
    
from sklearn.metrics import accuracy_score
print("Model accuracy score:{0:0.4f}".format(accuracy_score(y_test, y_pred)))


y_pred_train=knn.predict(x_train)
print("Trainning accuracy score:{0:0.4f}".format(accuracy_score(y_train,y_pred_train)))


# print the scores on training and test set

print('Training set score: {:.4f}'.format(knn.score(x_train, y_train)))

print('Test set score: {:.4f}'.format(knn.score(x_test, y_test)))


y_test.value_counts()


#check null acuuracy score:

null_accuracy=(85/(85+55))
print('Null accuracy score:{0:0.4f}'.format(null_accuracy))


# REBUILD KKN CLASSSIFICATION WITH DIFF K-VALUES:

knn_5=KNeighborsClassifier(n_neighbors=5) 
knn_5.fit(x_train, y_train)

y_pred_5=knn_5.predict(x_test)
print('Model accuracy score with k=5:{0:0.4f}'.format(accuracy_score(y_test, y_pred_5)))


# K=6:
    

knn_6=KNeighborsClassifier(n_neighbors=6) 
knn_6.fit(x_train, y_train)

y_pred_6=knn_6.predict(x_test)
print('Model accuracy score with k=6:{0:0.4f}'.format(accuracy_score(y_test, y_pred_6)))


#k=7:
    
knn_7=KNeighborsClassifier(n_neighbors=7) 
knn_7.fit(x_train, y_train)

y_pred_7=knn_7.predict(x_test)
print('Model accuracy score with k=7:{0:0.4f}'.format(accuracy_score(y_test, y_pred_7)))


# k=8:
    
knn_8=KNeighborsClassifier(n_neighbors=8) 
knn_8.fit(x_train, y_train)

y_pred_8=knn_8.predict(x_test)
print('Model accuracy score with k=8:{0:0.4f}'.format(accuracy_score(y_test, y_pred_8)))


#  k=9:
    
knn_9=KNeighborsClassifier(n_neighbors=9) 
knn_9.fit(x_train, y_train)

y_pred_9=knn_9.predict(x_test)
print('Model accuracy score with k=9:{0:0.4f}'.format(accuracy_score(y_test, y_pred_9)))


# confusion metrics:
    
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print(cm)
print('\nTrue Positives =', cm[1, 1])
print('\nTrue Negatives =', cm[0, 0])
print('\nFalse Positives =', cm[1, 0])
print('\nFalse Negatives =', cm[0, 1])


# for cm_7:
    
cm_7 = confusion_matrix(y_test, y_pred_7)
print(cm_7)
print('\nTrue Positives =', cm_7[1, 1])
print('\nTrue Negatives =', cm_7[0, 0])
print('\nFalse Positives =', cm_7[1, 0])
print('\nFalse Negatives =', cm_7[0, 1])


# visualization:
    
plt.figure(figsize=(6,4))
cm_matrix = pd.DataFrame(data=cm_7,columns=['Actual positives:1','Actual Negatives:0'],index=['Predict Positive:1','Predict Negative:0'])
sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')

#classification metrics:
    
    
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred_7))

TP = cm_7[1,1]
TN = cm_7[0,0]
FP = cm_7[1,0]
FN = cm_7[0,1]


classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)
classification_accuracy

classification_error = (FP + FN) / float(TP + TN + FP + FN)
classification_error


precision = TP / float(TP + FP)
precision

recall = TP / float(TP + FN)
recall

true_positive_rate = TP / float(TP + FN)
true_positive_rate

false_positive_rate = FP / float(FP + TN)
false_positive_rate

specificity = TN / (TN + FP)
specificity

y_pred_prob = knn.predict_proba(x_test)[0:10]
y_pred_prob


y_pred_prob_df = pd.DataFrame(data=y_pred_prob, columns=['Prob of - benign cancer (2)', 'Prob of - malignant cancer (4)'])
y_pred_prob_df



knn.predict_proba(x_test)[0:10, 1]


y_pred_1 = knn.predict_proba(x_test)[:, 1]

# histogram plot:
    
plt.figure(figsize=(6,4))
plt.rcParams['font.size'] = 12
plt.hist(y_pred_1, bins = 10)
plt.title('Histogram of predicted probabilities of malignant cancer')
plt.xlim(0,1)
plt.xlabel('Predicted probabilities of malignant cancer')
plt.ylabel('Frequency')


# ROC-AUC:
    

from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred, pos_label=4)
plt.figure(figsize=(6,4))

plt.plot(fpr, tpr, linewidth=2)
plt.plot([0,1], [0,1], 'k--' )
plt.rcParams['font.size'] = 12

plt.title('ROC curve for Breast Cancer kNN classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')

plt.show()

# compute ROC AUC:
    
from sklearn.metrics import roc_auc_score
ROC_AUC=roc_auc_score(y_test, y_pred)
ROC_AUC


# cross validation ROC AND AUC:

from sklearn.model_selection import cross_val_score
Cross_validated_ROC_AUC = cross_val_score(knn_7, x_train, y_train, cv=5, scoring='roc_auc').mean()
Cross_validated_ROC_AUC


#k-fold: cross validation:
    
from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn_7, x_train, y_train, cv = 10, scoring='accuracy')
scores

# compute Average cross-validation score

print('Average cross-validation score: {:.4f}'.format(scores.mean()))







