# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries.

2.Load the dataset and check for null data values and duplicate data values in the dataframe.

3.Import label encoder from sklearn.preprocessing to encode the dataset.

4.Apply Logistic Regression on to the model.

5.Predict the y values.

6.Calculate the Accuracy,Confusion and Classsification report.


## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: HAREVASU S
RegisterNumber:  212223230069
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv("Placement_Data.csv")
df
df.head()
df.tail()
df=df.drop(['sl_no','gender','salary'],axis=1)
df=df.drop(['ssc_b','hsc_b'],axis=1)
df.shape
df.info()
df["degree_t"]=df["degree_t"].astype("category")
df["hsc_s"]=df["hsc_s"].astype("category")
df["workex"]=df["workex"].astype("category")
df["status"]=df["status"].astype("category")
df["specialisation"]=df["specialisation"].astype("category")
df["degree_t"]=df["degree_t"].cat.codes
df["hsc_s"]=df["hsc_s"].cat.codes
df["workex"]=df["workex"].cat.codes
df["status"]=df["status"].cat.codes
df["specialisation"]=df["specialisation"].cat.codes
x=df.iloc[: ,:-1].values
y=df.iloc[:,- 1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

df.head()
from sklearn.linear_model import LogisticRegression

#printing its accuracy
clf=LogisticRegression()
clf.fit(x_train,y_train)
clf.score(x_test,y_test)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear") 
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = (y_test,y_pred)
confusion 

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
# Predicting for random value
clf.predict([[1	,78.33,	1,	2,	77.48,	2,	86.5,	0,	66.28]])
```

## Output:
![Screenshot 2024-09-21 091813](https://github.com/user-attachments/assets/d0a8f39b-8324-4c4b-b6e2-0bc8f528d6ae)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
