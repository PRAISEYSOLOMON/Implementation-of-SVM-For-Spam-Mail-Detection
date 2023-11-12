# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Read the data frame using pandas.
3. Get the information regarding the null values present in the dataframe.
4. Split the data into training and testing sets.
5. Convert the text data into a numerical representation using CountVectorizer.
6. Use a Support Vector Machine (SVM) to train a model on the training data and make predictions on the testing data.
7. Finally, evaluate the accuracy of the model.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: PRAISEY S
RegisterNumber:  212222040117

import chardet 
file='spam.csv'
with open(file, 'rb') as rawdata: 
    result = chardet.detect(rawdata.read(100000))
result
import pandas as pd
data = pd.read_csv("spam.csv",encoding="Windows-1252")
data.head()
data.info()
data.isnull().sum()

X = data["v1"].values
Y = data["v2"].values
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2, random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(X_train,Y_train)
Y_pred = svc.predict(X_test)
print("Y_prediction Value: ",Y_pred)

from sklearn import metrics
accuracy=metrics.accuracy_score(Y_test,Y_pred)
accuracy

*/
```

## Output:

Result Output

![out 9 1](https://github.com/PRAISEYSOLOMON/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119394259/340c9a3d-ee19-4acc-8551-3c6aee7a860e)

data.head()

![out 9 2](https://github.com/PRAISEYSOLOMON/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119394259/abe960ff-fe56-417d-9344-49ada6110841)

data.info()

![out 9 3](https://github.com/PRAISEYSOLOMON/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119394259/ae6c5157-48ac-44eb-9e92-569a2ea2e688)

data.isnull().sum()

![out 9 41](https://github.com/PRAISEYSOLOMON/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119394259/aa03fa82-43a8-454e-8a8a-c8b3bba9ab64)

![out 9 42](https://github.com/PRAISEYSOLOMON/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119394259/dec78e2e-0c70-4fc9-918c-a8425a53d5ee)

Y_prediction Value

![out 9 5](https://github.com/PRAISEYSOLOMON/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119394259/fc8d3169-52de-4908-a3f3-4966654fd21b)

Accuracy Value

![out 9 6](https://github.com/PRAISEYSOLOMON/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119394259/c8fd6381-999e-4bae-aedc-78988d75bfc3)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
