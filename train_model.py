#import the libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib
#Load the dataset
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")
#Simple preprocessing
df = df[['Sex','Fare','Survived']].dropna()
df['Sex'] = df['Sex'].map({'male':0,'female':1})
X = df[['Sex','Fare']]
y = df['Survived']
#Train the model and dump it in pickle file
model = LogisticRegression()
model.fit(X,y)
joblib.dump(model,'titanic_model.pkl')