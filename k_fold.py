import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv(r"C:\Users\Lenovo\Documents\Notes\APRIL\22nd- Cross validation\1.K-FOLD CROSS VALIDATION CODE_ MODEL SELECTION\Social_Network_Ads.csv")
x = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)


from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)
print(ac)

bias = classifier.score(x_train, y_train)
bias

variance = classifier.score(x_test, y_test)
variance


# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train, cv = 5)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
#print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
