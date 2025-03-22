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

/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: MUTHULAKSHMI D
RegisterNumber:  212223040122
*/
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

![Screenshot 2025-03-22 131543](https://github.com/user-attachments/assets/0a81a526-f6fb-47f8-bd89-2f39e34a68d8)
![Screenshot 2025-03-22 131557](https://github.com/user-attachments/assets/4ea9bee2-3c6c-4cfc-a855-fb21c7f9b5f6)
![Screenshot 2025-03-22 131613](https://github.com/user-attachments/assets/464cbf9f-57c7-4aec-a552-b7966d7f2f6b)
![Screenshot 2025-03-22 131622](https://github.com/user-attachments/assets/f260df13-8ad9-479f-9867-bdea536dbc09)
![Screenshot 2025-03-22 131448](https://github.com/user-attachments/assets/a49638df-0ef3-486c-be1f-28912847ecf1)
![Screenshot 2025-03-22 131455](https://github.com/user-attachments/assets/86c2cc14-e000-4e36-bcf7-31272e91225a)
![Screenshot 2025-03-22 131502](https://github.com/user-attachments/assets/813dae71-dd9d-4f95-a26a-309f6dddf114)
![Screenshot 2025-03-22 131508](https://github.com/user-attachments/assets/8673fd0e-b7f2-4d6f-9c46-e7167be133a7)
![Screenshot 2025-03-22 131513](https://github.com/user-attachments/assets/662c54ef-3e7c-4648-8666-df31b3bbf7d2)


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
