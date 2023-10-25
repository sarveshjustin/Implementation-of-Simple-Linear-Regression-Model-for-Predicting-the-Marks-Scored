# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
### 1.Import the standard Libraries.
### 2.Set variables for assigning dataset values.
### 3.Import linear regression from sklearn.
### 4.Assign the points for representing in the graph.
### 5.Predict the regression for marks by using the representation of the graph.
### 6.Compare the graphs and hence we obtained the linear regression for the given datas.
## Program:
```
#Program to implement the simple linear regression model for predicting the marks scored.
#Developed by: sarvesh.s
#RegisterNumber: 212222230135

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
### Dataset
![image](https://github.com/Afsarjumail/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118343395/45fa3abc-3b5c-4053-9552-55c8f5f8f8da)


### Head Values
![image](https://github.com/Afsarjumail/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118343395/1a2c1afc-5371-49db-b70a-6cda3275d251)


### Tail Values
![image](https://github.com/Afsarjumail/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118343395/131d50d8-8c19-483f-9730-e7906f538cab)


### X and Y values
![image](https://github.com/Afsarjumail/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118343395/306fa17d-a2e1-490e-9fcd-e9f096687bcb)


### Predication values of X and Y
![image](https://github.com/Afsarjumail/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118343395/ebea550e-caf3-4123-b568-6b28b782e1e6)

### MSE,MAE and RMSE
![image](https://github.com/Afsarjumail/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118343395/bfeaaba9-67dc-4bd4-b115-c415495b9d75)

### Training Set
![image](https://github.com/Afsarjumail/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118343395/80547da3-660a-4342-ad80-4bff055873d7)


### Testing Set
![image](https://github.com/Afsarjumail/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/118343395/57ef8870-3a61-4a19-8590-b71d5f29fae8)



### Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.