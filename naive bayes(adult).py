import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns


df=pd.read_csv(r"C:\Users\Lenovo\Documents\Notes\APRIL\22nd- Cross validation\adult.csv")
df.head()


df.shape
df.describe()
df.info()
df.isnull().sum()

col_names= ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship',
            'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']

df.columns=col_names
df.columns


df.head()
df.info()


# find categorical variables
categorical = [var for var in df.columns if df[var].dtype=='O']
print('There are {} categorical variables\n'.format(len(categorical)))
print('The categorical variables are :\n\n', categorical)


df[categorical].head()
df[categorical].isnull().sum()


# view frequency counts of values in categorical variables
for var in categorical: 
    print(df[var].value_counts())


# distribution of frequency of catgorical variable:
    

for var in categorical: 
    print(df[var].value_counts()/float(len(df)))



df.workclass.unique()

df.workclass.value_counts()

df['workclass'].replace('?', np.NaN, inplace=True)
df.workclass.value_counts()


df.occupation.unique()
df.occupation.value_counts()
df['workclass'].replace('?', np.NaN, inplace=True)
df.occupation.value_counts()


df.native_country.unique()
df.native_country.value_counts()
df['native_country'].replace('?', np.NaN, inplace=True)
df.native_country.value_counts()


df[categorical].isnull().sum()


for var in categorical:
    
    print(var, ' contains ', len(df[var].unique()), ' labels')

numerical = [var for var in df.columns if df[var].dtype!='O']

print('There are {} numerical variables\n'.format(len(numerical)))

print('The numerical variables are :', numerical)

# view the numerical variables

df[numerical].head()

df[numerical].isnull().sum()

#x and y split:
x=df.drop(['income'],axis=1)
y=df['income']


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

x_train.shape
x_test.shape

y_train.shape
y_test.shape

x_train.dtypes

# to make categoricsal to Zero :

categorical = [col for col in x_train.columns if x_train[col].dtypes == 'O']
categorical

numerical = [col for col in x_train.columns if x_train[col].dtypes != 'O']
numerical

x_train[categorical].isnull().mean()


for col in categorical:
    if x_train[col].isnull().mean()>0:
       print(col, (x_train[col].isnull().mean()))
        


x_train[categorical].isnull().sum()
x_test[categorical].isnull().sum()

x_train.isnull().sum()
x_test.isnull().sum()

categorical

x_train[categorical].head()


# import category encoders
import category_encoders as ce

# encode remaining variables with one-hot encoding:
    
encoder = ce.OneHotEncoder(cols=['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country'])

x_train = encoder.fit_transform(x_train)
x_test = encoder.transform(x_test)


x_train.head()
x_train.shape


x_test.head()
x_test.shape


cols=x_train.columns

from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)

x_train=pd.DataFrame(x_train,columns=[cols])
x_test=pd.DataFrame(x_test,columns=[cols])

x_train.head()


from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(x_train,y_train)

y_pred=gnb.predict(x_test)
y_pred


from sklearn.metrics import accuracy_score
print('Model accuracy score:{0:0.4f}'.format(accuracy_score(y_test, y_pred)))


y_pred_train=gnb.predict(x_train)
y_pred_train


print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train, y_pred_train)))


# scores :
    
print('Training set score: {:.4f}'.format(gnb.score(x_train, y_train)))
print('Test set score: {:.4f}'.format(gnb.score(x_test, y_test)))

y_test.value_counts()
null_accuracy=  (7454/( 7454+2314))
print(null_accuracy)


# confusion matrix:
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)
print('\nTrue Positives(TP) = ', cm[1,1])
print('\nTrue Negatives(TN) = ', cm[0,0])
print('\nFalse Positives(FP) = ', cm[1,0])
print('\nFalse Negatives(FN) = ', cm[0,1])    


#visual with heatmap:
    
cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'], 
                                 index=['Predict Positive:1', 'Predict Negative:0'])

sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')



from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

TP = cm[1,1]
TN = cm[0,0]
FP = cm[1,0]
FN = cm[0,1]


classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)
print(classification_accuracy)

classification_error = (FP + FN) / float(TP + TN + FP + FN)
print(classification_error)


precision = TP / float(TP + FP)
print(precision)

recall= TP/float(TP+FN)
print(recall)

true_positive_rate = TP / float(TP + FN)
true_positive_rate

false_positive_rate = FP / float(FP + TN)
false_positive_rate

specificity = TN / (TN + FP)
print(specificity)


# calcculating probabilities:
    
y_pred_prob=gnb.predict(x_test)[0:10]
y_pred_prob


y_pred_prob_df = pd.DataFrame(data=y_pred_prob, columns=['Prob of - <=50K', 'Prob of - >50K'])
y_pred_prob_df


gnb.predict_proba(x_test)[0:10, 1]
y_pred1 = gnb.predict_proba(x_test)[:, 1]

# adjust the font size 
plt.rcParams['font.size'] = 12
plt.hist(y_pred1, bins = 10)
plt.title('Histogram of predicted probabilities of salaries >50K')
plt.xlim(0,1)
plt.xlabel('Predicted probabilities of salaries >50K')
plt.ylabel('Frequency')

#roc:

from sklearn.metrics import roc_curve
fpr,tpr,thresholds=roc_curve(y_test,y_pred1,pos_label='>50k')
plt.figure(figsize=(6,4))
plt.plot(fpr,tpr,linewidth=2)
plt.plot([0,1],[0,1],'k--')
plt.rcParams['font.size']=12
plt.title('ROC curve for Gaussian Naive Bayes Classifier for Predicting Salaries')

plt.xlabel('FPR(1-Specificity)')
plt.ylabel("TPR(Sensitivity)")
plt.show()


from sklearn.metrics import roc_auc_score
ROC_AUC=roc_auc_score(y_test,y_pred1)

ROC_AUC


# K_FOLD:

from sklearn.model_selection import cross_val_score

scores = cross_val_score(gnb, x_train, y_train, cv = 10, scoring='accuracy')

scores


print('Average cross-validation score: {:.4f}'.format(scores.mean()))



