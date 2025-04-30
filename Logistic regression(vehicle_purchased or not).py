import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv(r"C:\Users\Lenovo\Documents\Notes\APRIL\logit classification.csv")
df

x=df.iloc[:,[2,3]].values
y=df.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0) 
# rs ahs zero acuuracy will be more compare to all the rs=100,41,51

from sklearn.preprocessing import StandardScaler  #Normalizer the accuracy and the bais ,variance as less so we wont us the model
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

from sklearn.linear_model import LogisticRegression
lg_classifier = LogisticRegression()
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

#predicting the future  mine===>code
#future_pred=pd.read_csv(r"C:\Users\Lenovo\Documents\Notes\APRIL\final1.csv")
#fp=future_pred.iloc[:,[3,4]].values
#fp=sc.fit_transform(fp)
#future=lg_classifier.predict(fp)
#future


# future prediction:
    
dataset1=pd.read_csv(r"C:\Users\Lenovo\Documents\Notes\APRIL\9th -15. Logistic regression with future prediction\Future prediction1.csv")
d2=dataset1.copy()
dataset1=dataset1.iloc[:,[2,3]].values # need to last to attributes

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()  # to convert data to scaling part
M=sc.fit_transform(dataset1)

y_pred1=pd.DataFrame()

d2['y_pred1']=lg_classifier.predict(M)
d2.to_csv('pred_model.csv')

# to get the path
import os 
os.getcwd()






