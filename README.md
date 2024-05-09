<h3>Date:</h3>
<h2>Exp: 07</h2>

# Implementation of Decision Tree Regressor Model for Predicting the Salary of the Employee
## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook
## Algorithm
1. import libraries
2. read csv file
3. Find the values for MSE MAE and R-SQUARE
4. Print and end the program
## Program:
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

Developed by: Sabari Akash A
RegisterNumber:  212222230124
```py
import pandas as pd
data=pd.read_csv("/content/Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data['Position']=le.fit_transform(data['Position'])
data.head()
x=data[['Position','Level']]
x
y=data['Salary']
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeClassifier,plot_tree
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
import matplotlib.pyplot as plt
dt.predict([[5,6]])
plt.figure(figsize=(20,8))
plot_tree(dt,feature_names=x.columns,filled=True)
plt.show()
```
## Output:
<img src=image.png width=300 height=300>
<img src=image-1.png width=300 height=300>
<img src=image-2.png width=200 height=100>
<img src=image-3.png width=200 height=300>
<img src=image-4.png width=200 height=100>
<img src=image-5.png width=200 height=100>
<br>

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
