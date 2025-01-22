import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

data = {
    'Age': [22, np.nan, 24, 30, 25],
    'Salary': [50000, 60000, np.nan, 70000, 65000],
    'Country': ['USA', 'Canada', 'USA', 'Canada', np.nan],
    'Gender': ['Male', 'Female', 'Female', 'Male', 'Female']
},

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

imputer = SimpleImputer(strategy='mean')
df[['Age', 'Salary']] = imputer.fit_transform(df[['Age', 'Salary']])

imputer_cat = SimpleImputer(strategy='most_frequent')
df[['Country']] = imputer_cat.fit_transform(df[['Country']])

print("\nDataFrame after handling missing values:")
print(df)

label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['Country'] = label_encoder.fit_transform(df['Country'])

print("\nDataFrame after encoding categorical variables:")
print(df)

scaler = StandardScaler()
df[['Age', 'Salary']] = scaler.fit_transform(df[['Age', 'Salary']])

print("\nDataFrame after feature scaling:")
print(df)