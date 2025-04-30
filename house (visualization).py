import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
#np.set_printoptions(threshold=np.nan)
np.set_printoptions(threshold=sys.maxsize)

house=pd.read_csv(r"C:\Users\Lenovo\Documents\Notes\MARCH\20th- slr\SLR - House price prediction\House_data.csv")
space=house['sqft_living']
price=house['price']

x=np.array(space).reshape(-1,1)
y=np.array(price)

#split
#from sklearn.cross_Validation import train_test_split  (update cross_validation -->> model_selection)
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.33,random_state=0)

#simple LR to train the set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(xtrain,ytrain)
# predict price
pred=regressor.predict(xtest)


#visualization(train result)
plt.scatter(xtrain,ytrain,color='red')
plt.plot(xtrain,regressor.predict(xtrain),color='blue')
plt.title("Visuals for Training Datset")
plt.xlabel('Space')
plt.ylabel('Price')
plt.show()  # Ensure you call plt.show() to display the plot


#visualization(test)
plt.scatter(xtest,ytest,color='red')
plt.plot(xtrain,regressor.predict(xtrain),color='blue')
plt.title("Visuals for Test Dataset")
plt.xlabel('Space')
plt.ylabel('Price')
plt.show()
