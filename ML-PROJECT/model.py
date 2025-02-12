import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv('house_data.csv')  


X = data[['sqft_living']]  
y = data['price'] 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = LinearRegression()
model.fit(X_train, y_train)


with open('house_data.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved as 'house_price_model.pkl'")
