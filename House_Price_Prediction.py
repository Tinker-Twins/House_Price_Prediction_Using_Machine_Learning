import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import os
path = os.getcwd() + '\Data\House_Price_Training_Data.csv'
org_data = pd.read_csv(path)
org_data.head()

org_data.plot(kind='scatter', x='Size', y='Price', figsize=(10,5) )

data = (org_data - org_data.mean())/(org_data.max() - org_data.min())
data.head()

data.plot(kind='scatter', x='Size', y='Price', figsize=(10,5) )

data.shape

data.insert (0, 'Ones', 1)
data.head()

cols = data.shape[1]
print (cols)

x = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]

x = np.matrix(x)
y = np.matrix(y)
theta = np.matrix(np.array([0,0]))

x.shape, theta.shape, y.shape

def computeError(x, y, theta):
    inner = np.power(((x*theta.T)-y), 2)
    return np.sum(inner)/(2*len(x))

computeError(x, y, theta)

Learn_rate=1
iters=150
def gradientDescent(x,y,theta,Learn_Rate,iters):
    temp=np.matrix(np.zeros(theta.shape))
    parameters=int(theta.ravel().shape[1])
    cost=np.zeros(iters)
    
    for i in range(iters):
        error=(x*theta.T)-y
        
        for j in range(parameters):
            term=np.multiply(error,x[:,j])
            temp[0,j]=theta[0,j]-((Learn_rate/len(x))*np.sum(term))
            
        theta=temp
        cost[i]=computeError(x,y,theta)
        
    return theta,cost

g,cost=gradientDescent(x,y,theta,Learn_rate,iters)
print(g,cost)

x=data.Size

f=g[0,0]+(g[0,1]*x)

fig,ax=plt.subplots(figsize=(12,8))
ax.plot(x,f,'r',label='Prediction')
ax.scatter(data.Size,data.Price,label='Training Data')
ax.legend(loc=2)
ax.set_xlabel('Size')
ax.set_ylabel('Price')
ax.set_title('Predicted Price vs. Size')

fig,ax=plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters),cost,'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Error')
ax.set_title('Error vs. Iterations')

from sklearn.metrics import r2_score
acc=(r2_score(y,f))
print("Accuracy = {}%".format(acc*100))

def predict(theta,acc):
    #Get input from user
    size=float(input("Enter the size of the house: "))
    #Mean normalization
    size=(size-org_data.Size.mean())/(org_data.Size.max()-org_data.Size.min())
    #Model
    price=(theta[0,0]+(theta[0,1]*size))
    #Reversing mean normalization
    new_price=(price*(org_data.Price.max()-org_data.Price.min())+(org_data.Price.mean()))
    
    price_at_max_acc=(new_price*(1/acc))
    price_range=price_at_max_acc-new_price
    
    return new_price, price_range

New_price,price_range=predict(g,acc)
print("The cost of your house will be "+str(New_price)+' Lakhs'+' (+ or -) '+str(price_range)+' Lakhs')