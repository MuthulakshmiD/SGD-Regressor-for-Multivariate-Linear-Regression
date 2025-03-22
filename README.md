# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load California housing data, select three features as X and two target variables as Y, then split into train and test sets.
2. Standardize X and Y using StandardScaler for consistent scaling across features.
3. Initialize SGDRegressor and wrap it with MultiOutputRegressor to handle multiple targets.
4. Train the model on the standardized training data.
5. Predict on the test data, inverse-transform predictions, compute mean squared error, and print results. 

## Program:


Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: MUTHULAKSHMI D
RegisterNumber:  212223040122
```
# exp-3
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

dataset=fetch_california_housing()
df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
df['HousingPrice']=dataset.target
df.head()

df.info()

x=df.drop(columns=['AveOccup','HousingPrice'])
x.info()

y=df[['AveOccup','HousingPrice']]
y.info()

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

scaler_x=StandardScaler()
scaler_y=StandardScaler()
x_train=scaler_x.fit_transform(x_train)
y_train=scaler_y.fit_transform(y_train)
x_test=scaler_x.transform(x_test)
y_test=scaler_y.transform(y_test)
print(x_train)
print(y_train)
print(x_test)
print(y_test)

sgd=SGDRegressor(max_iter=1000,tol=1e-3)
multi_output_sgd=MultiOutputRegressor(sgd)
multi_output_sgd.fit(x_train,y_train)

y_pred=multi_output_sgd.predict(x_test)
y_pred=scaler_y.inverse_transform(y_pred)
y_test=scaler_y.inverse_transform(y_test)
mse=mean_squared_error(y_test,y_pred)
print("Mean Squared Error :",mse)
print(y_pred)
```
## Output:
# df.head():
![Screenshot 2025-03-22 131448](https://github.com/user-attachments/assets/ccca5444-73dd-4d15-b05b-41028b42f798)

# df.info():
![Screenshot 2025-03-22 131455](https://github.com/user-attachments/assets/5c05a082-aa3b-41c3-a46f-561cc2304077)

# x.info():
![Screenshot 2025-03-22 131502](https://github.com/user-attachments/assets/dfd85a6a-e531-4bd4-b088-fb815c95323b)

# y.info():
![Screenshot 2025-03-22 131508](https://github.com/user-attachments/assets/4483f4ad-8ea1-4c95-ba95-a06e593ebdfa)

# shape of X & y:
![Screenshot 2025-03-22 131513](https://github.com/user-attachments/assets/a51eeebd-7fb8-408f-98c0-3c0e946a8ce5)

# x & y:
![Screenshot 2025-03-22 131543](https://github.com/user-attachments/assets/a8c661d8-dc18-4aac-906d-9e72d290af75)

![Screenshot 2025-03-22 131557](https://github.com/user-attachments/assets/a1e36426-cf86-4dda-8e8f-a08b5eaee562)


# MultioutputRegressor:
![Screenshot 2025-03-22 131613](https://github.com/user-attachments/assets/1b31b3f0-b4f4-4355-8694-d860cc9b67a7)

# y_pred & Mean Squared Error:
![Screenshot 2025-03-22 131622](https://github.com/user-attachments/assets/ed07ba92-0e34-463c-8663-2996b0410a92)


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
