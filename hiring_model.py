import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pickle

data = pd.read_csv("data/hiring.csv", delimiter=",")

#replacing missing values
data["experience"].fillna(0, inplace=True)
data["test_score"].fillna(data["test_score"].mean(), inplace=True)

def convert_to_int(word):
    word_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 
     'seven':7, 'eight':8, 'nine':'9', 'ten':10, 'eleven':11, 
     'twelve':12, 'zero':0, 0:0}
    return word_dict[word]

data["experience"] = data["experience"].apply(lambda x : convert_to_int(x))

x = data.iloc[:, :3]
y = data.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

reg = LinearRegression()
reg.fit(x_train, y_train)
y_pred = reg.predict(x_test)

print(r2_score(y_test,y_pred))
print(mean_absolute_error(y_test,y_pred))
print(mean_squared_error(y_test,y_pred))

pickle.dump(reg, open("hiring_model.pkl", "wb"))

model = pickle.load(open("hiring_model.pkl", "rb"))
output = model.predict([[2,7,9]])
print(output)
