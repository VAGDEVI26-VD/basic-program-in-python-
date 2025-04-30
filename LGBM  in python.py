import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv(r"C:\Users\Lenovo\Documents\Notes\APRIL\22nd- Cross validation\Breast_cancer_data.csv")
data.head()

data.shape
data.isnull().sum()
data.describe()
data.info()


data['diagnosis'].value_counts()

x=data[['mean_radius','mean_texture','mean_perimeter','mean_area','mean_smoothness']]
y=data['diagnosis']


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

# LGBM MODEL:
    

import lightgbm as lgb
clf = lgb.LGBMClassifier()
clf.fit(x_train, y_train)

y_pred=clf.predict(x_test)


from sklearn.metrics import accuracy_score
ac=accuracy_score(y_test, y_pred)
ac

#train ac:

y_pred_train = clf.predict(x_train)
y_pred_train



# check for overfitting:

print('Training set score: {:.4f}'.format(clf.score(x_train, y_train)))
print('Test set score: {:.4f}'.format(clf.score(x_test, y_test)))


# cm:
    
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pred)
cm


print('TP =',cm[1,1])
print('TN =',cm[0,0])
print('FP =',cm[0,1])
print('FN =',cm[1,0])


# visualize confusion matrix with seaborn heatmap:
    
cm_matrix = pd.DataFrame(data=cm, columns=['Actual Positive:1', 'Actual Negative:0'],index=['Predict Positive:1', 'Predict Negative:0'])
sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')


from sklearn.metrics import classification_report
cr=classification_report(y_test, y_pred)
cr
