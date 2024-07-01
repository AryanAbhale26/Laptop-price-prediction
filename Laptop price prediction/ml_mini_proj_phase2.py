# -*- coding: utf-8 -*-
"""ML_mini_proj phase2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1TSqONFyqEryU_sq5mrPn5Ppn49IZ83GF
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/content/laptop_data (1).csv')

"""# **Data Cleaning**"""

df.head()

df.shape

df.info()

df.duplicated().sum()

"""AS there are no null values we don't have to handle them"""

df.isnull().sum()

"""**Droping Unamed column**"""

df.drop(columns=['Unnamed: 0'],inplace=True)

df.head()

df['Ram'] = df['Ram'].str.replace('GB','')
df['Weight'] = df['Weight'].str.replace('kg','')

df.head()

"""**Converting the columns in required data types**"""

df['Ram'] = df['Ram'].astype('int32')
df['Weight'] = df['Weight'].astype('float32')



df.info()

"""# **EDA**"""

import seaborn as sns

"""As we can see that in our data is skewed"""

sns.distplot(df['Price'])

"""Most laptops are of Dell and Lenovo"""

df['Company'].value_counts().plot(kind='bar')

"""Comparing the price of the laptops

Here the Razer brand has highest price of laptops
"""

sns.barplot(x=df['Company'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()

"""Comparing the type of laptops are there in dataset"""

df['TypeName'].value_counts().plot(kind='bar')

"""Comparing the price of laptops according to their type



*   Here the workstaion laptop's are of high price



"""

sns.barplot(x=df['TypeName'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()

"""Checking the frequency size of laptops  available in the data set


*   Here most no.of laptops lie between 15-16 inches



"""

sns.distplot(df['Inches'])

"""Comparing the price of laptops according to Size of laptop"""

sns.scatterplot(x=df['Inches'],y=df['Price'])

"""checking the Frequency of laptops  according to sccreen Resolution

*   In the dataset most no.of laptops are of Full HD 1920x1080


"""

df['ScreenResolution'].value_counts()

"""------------------------------------------------------------
Making a column of touchscreen  in which has values weather the laptops are touch screen or not
"""

df['Touchscreen'] = df['ScreenResolution'].apply(lambda x:1 if 'Touchscreen' in x else 0)

df.sample(5)

df['Touchscreen'].value_counts().plot(kind='bar')

sns.barplot(x=df['Touchscreen'],y=df['Price'])

"""Making a column to weather the laptops screen has ips panel or not"""

df['Ips'] = df['ScreenResolution'].apply(lambda x:1 if 'IPS' in x else 0)

df.head()

df['Ips'].value_counts().plot(kind='bar')

sns.barplot(x=df['Ips'],y=df['Price'])

"""Spliting the Resolution values in two different columns for prediction"""

new = df['ScreenResolution'].str.split('x',n=1,expand=True)

df['X_res'] = new[0]
df['Y_res'] = new[1]

df.sample(5)

df['X_res'] = df['X_res'].str.replace(',','').str.findall(r'(\d+\.?\d+)').apply(lambda x:x[0])

df.head()

df['X_res'] = df['X_res'].astype('int')
df['Y_res'] = df['Y_res'].astype('int')

df.info()

"""---------------------------------------------------------------------
Checking the correlation with price

*   As we can see that it is less correlated to inches  column and weight


"""

numeric_columns = ['Inches', 'Ram', 'Weight', 'Touchscreen', 'Ips', 'X_res', 'Y_res']

correlation_with_price = df[numeric_columns].corrwith(df['Price'])

print("Correlation of numeric columns with respect to Price:")
print(correlation_with_price)

"""Creating a new column ppi(pixel per inch)"""

df['ppi'] = (((df['X_res']**2) + (df['Y_res']**2))**0.5/df['Inches']).astype('float')

df.drop(columns=['ScreenResolution'],inplace=True)

df.head()

df.drop(columns=['Inches','X_res','Y_res'],inplace=True)

df.head()

df.head()

df.head()

"""Checking in terms of RAM"""

df['Ram'].value_counts().plot(kind='bar')

"""Here we can see higher the RAM higher the price"""

sns.barplot(x=df['Ram'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()

"""Checking the Memory column"""

df['Memory'].value_counts()

"""Making column for each type of memory"""

print(df['Memory'].unique())

import pandas as pd

# Assuming df is your DataFrame containing the 'Memory' column

# Convert Memory column to strings and remove unnecessary text
df['Memory'] = df['Memory'].astype(str).replace('\.0', '', regex=True)
df["Memory"] = df["Memory"].str.replace('GB', '')
df["Memory"] = df["Memory"].str.replace('TB', '000')

# Split Memory into two parts
new = df["Memory"].str.split("+", n=1, expand=True)
df["first"] = new[0].str.strip()
df["second"] = new[1].fillna("0")

# Extract numerical values from first and second parts
df['first'] = df['first'].str.extract('(\d+)')
df['second'] = df['second'].str.extract('(\d+)')

# Fill NaN values with 0
df['second'].fillna('0', inplace=True)

# Convert numerical parts to integers
df['first'] = df['first'].astype(int)
df['second'] = df['second'].astype(int)

# Define a function to calculate memory based on type
def calculate_memory(row, memory_type):
    if memory_type in row['Memory']:
        return row['first'] + row['second']
    else:
        return 0

# Calculate total storage capacity for each type
df['HDD'] = df.apply(lambda row: calculate_memory(row, 'HDD'), axis=1)
df['SSD'] = df.apply(lambda row: calculate_memory(row, 'SSD'), axis=1)
df['Hybrid'] = df.apply(lambda row: calculate_memory(row, 'Hybrid'), axis=1)
df['Flash_Storage'] = df.apply(lambda row: calculate_memory(row, 'Flash Storage'), axis=1)

# Now you can proceed with your calculations and data manipulation

df.sample(5)

df.drop(columns=['Memory'],inplace=True)

df.info()

import pandas as pd

numeric_columns = ['Ram', 'Weight', 'Touchscreen', 'Ips', 'first', 'second', 'HDD', 'SSD', 'Hybrid', 'Flash_Storage']

correlation_with_price = df[numeric_columns].corrwith(df['Price'])

print("Correlation of numeric columns with respect to Price:")
print(correlation_with_price)

df.drop(columns=['Hybrid','Flash_Storage'],inplace=True)

df.head()

"""Ckecking for GPU"""

df['Gpu'].value_counts()

df['Gpu brand'] = df['Gpu'].apply(lambda x:x.split()[0])

df.head()

df['Gpu brand'].value_counts()

df = df[df['Gpu brand'] != 'ARM']

df['Gpu brand'].value_counts()

"""Here we can see that GPU of nvidia has more price"""

sns.barplot(x=df['Gpu brand'],y=df['Price'],estimator=np.median)
plt.xticks(rotation='vertical')
plt.show()

df.drop(columns=['Gpu'],inplace=True)

df.head()

"""Checking for Operating system"""

df['OpSys'].value_counts()

"""Here we can see that Price of mac and windows 7 is more as compared to others

"""

sns.barplot(x=df['OpSys'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()

"""Function for extracting the Os WINDOWS,MAC and others"""

def cat_os(inp):
    if inp == 'Windows 10' or inp == 'Windows 7' or inp == 'Windows 10 S':
        return 'Windows'
    elif inp == 'macOS' or inp == 'Mac OS X':
        return 'Mac'
    else:
        return 'Others/No OS/Linux'

df['os'] = df['OpSys'].apply(cat_os)

df.head()

df.drop(columns=['OpSys'],inplace=True)

"""Price of Mac is higher than windows"""

sns.barplot(x=df['os'],y=df['Price'])
plt.xticks(rotation='vertical')
plt.show()

sns.distplot(df['Weight'])

"""  We can see that there is no correlation of price with weight


"""

sns.scatterplot(x=df['Weight'],y=df['Price'])

df.info()

import pandas as pd

numeric_columns = ['Ram', 'Weight', 'Price', 'Touchscreen', 'Ips', 'ppi', 'HDD', 'SSD']

correlation_with_price = df[numeric_columns].corrwith(df['Price'])

print("Correlation of numeric columns with respect to Price:")
print(correlation_with_price)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
numeric_columns = ['Ram', 'Weight', 'Price', 'Touchscreen', 'Ips', 'ppi', 'HDD', 'SSD']
correlation_matrix = df[numeric_columns].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap')
plt.show()

sns.distplot(df['Price'])

"""By taking the log of price column we can see that we are getting symmetric graph/normal distibuted graph"""

sns.distplot(np.log(df['Price']))

"""# **Train Test split**"""

X = df.drop(columns=['Price'])
y = np.log(df['Price'])

X

y

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.15,random_state=2)

X_train

"""Importing required libs"""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import r2_score,mean_absolute_error

from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor,ExtraTreesRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

"""#Linear regression"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify numeric and non-numeric columns
numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
non_numeric_cols = X_train.select_dtypes(exclude=np.number).columns.tolist()

# Define the preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(sparse=False, drop='first', handle_unknown='ignore'), non_numeric_cols),  # Assuming these are categorical columns
        ('num', SimpleImputer(strategy='mean'), numeric_cols)  # Impute missing values for numeric columns only
    ],
    remainder='passthrough'
)

# Define the pipeline
pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Fit the pipeline
pipe.fit(X_train, y_train)

# Predictions
y_pred = pipe.predict(X_test)

# Evaluate your model
print('R2 score:', r2_score(y_test, y_pred))
print('MAE:', mean_absolute_error(y_test, y_pred))

"""# Ridge regression"""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
non_numeric_cols = X_train.select_dtypes(exclude=np.number).columns.tolist()

# Define the preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(sparse=False, drop='first', handle_unknown='ignore'), non_numeric_cols),  # Assuming these are categorical columns
        ('num', SimpleImputer(strategy='mean'), numeric_cols)  # Impute missing values for numeric columns only
    ],
    remainder='passthrough'
)

# Define the pipeline
pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', Ridge(alpha=1.0))  # You can adjust the alpha parameter as needed
])

# Fit the pipeline
pipe.fit(X_train, y_train)

# Predictions
y_pred = pipe.predict(X_test)

# Evaluate your model
print('R2 score:', r2_score(y_test, y_pred))
print('MAE:', mean_absolute_error(y_test, y_pred))

"""# Lasso regression"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify numeric and non-numeric columns
numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
non_numeric_cols = X_train.select_dtypes(exclude=np.number).columns.tolist()

# Define the preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(sparse=False, drop='first', handle_unknown='ignore'), non_numeric_cols),  # Assuming these are categorical columns
        ('num', SimpleImputer(strategy='mean'), numeric_cols)  # Impute missing values for numeric columns only
    ],
    remainder='passthrough'
)

# Define the pipeline
pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', Lasso(alpha=1.0))  # You can adjust the alpha parameter as needed
])

# Fit the pipeline
pipe.fit(X_train, y_train)

# Predictions
y_pred = pipe.predict(X_test)

# Evaluate your model
print('R2 score:', r2_score(y_test, y_pred))
print('MAE:', mean_absolute_error(y_test, y_pred))

"""#Decision tree"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify numeric and non-numeric columns
numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
non_numeric_cols = X_train.select_dtypes(exclude=np.number).columns.tolist()

# Define the preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(sparse=False, drop='first', handle_unknown='ignore'), non_numeric_cols),  # Assuming these are categorical columns
    ],
    remainder='passthrough'
)

# Define the pipeline
pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', DecisionTreeRegressor())  # You can set hyperparameters of DecisionTreeRegressor here if needed
])

# Fit the pipeline
pipe.fit(X_train, y_train)

# Predictions
y_pred = pipe.predict(X_test)

# Evaluate your model
print('R2 score:', r2_score(y_test, y_pred))
print('MAE:', mean_absolute_error(y_test, y_pred))

"""Using SVM and Rbf kernel"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify numeric and non-numeric columns
numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
non_numeric_cols = X_train.select_dtypes(exclude=np.number).columns.tolist()

# Define the preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), numeric_cols),
        ('cat', OneHotEncoder(sparse=False, handle_unknown='ignore'), non_numeric_cols)
    ],
    remainder='passthrough'
)

# Define the SVR model
step2 = SVR(kernel='rbf', C=10000, epsilon=0.1)

# Define the pipeline
pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', step2)
])

# Fit the pipeline
pipe.fit(X_train, y_train)

# Predictions
y_pred = pipe.predict(X_test)

# Evaluate your model
print('R2 score:', r2_score(y_test, y_pred))
print('MAE:', mean_absolute_error(y_test, y_pred))

"""#Random forest"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify numeric and non-numeric columns
numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
non_numeric_cols = X_train.select_dtypes(exclude=np.number).columns.tolist()

# Define the preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(sparse=False, handle_unknown='ignore'), non_numeric_cols),
        ('num', SimpleImputer(strategy='mean'), numeric_cols)
    ],
    remainder='passthrough'
)

# Define the RandomForestRegressor model
step2 = RandomForestRegressor(n_estimators=100,
                              random_state=3,
                              max_samples=0.5,
                              max_features=0.75,
                              max_depth=15)

# Define the pipeline
pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', step2)
])

# Fit the pipeline
pipe.fit(X_train, y_train)

# Predictions
y_pred = pipe.predict(X_test)

# Evaluate your model
print('R2 score:', r2_score(y_test, y_pred))
print('MAE:', mean_absolute_error(y_test, y_pred))

"""# AdaBoost"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify numeric and non-numeric columns
numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
non_numeric_cols = X_train.select_dtypes(exclude=np.number).columns.tolist()

# Define the preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(sparse=False, handle_unknown='ignore'), non_numeric_cols),
        ('num', SimpleImputer(strategy='mean'), numeric_cols)
    ],
    remainder='passthrough'
)

# Define the AdaBoostRegressor model
step2 = AdaBoostRegressor(n_estimators=100, random_state=42)

# Define the pipeline
pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', step2)
])

# Fit the pipeline
pipe.fit(X_train, y_train)

# Predictions
y_pred = pipe.predict(X_test)

# Evaluate your model
print('R2 score:', r2_score(y_test, y_pred))
print('MAE:', mean_absolute_error(y_test, y_pred))

"""#Gradient Boost"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify numeric and non-numeric columns
numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
non_numeric_cols = X_train.select_dtypes(exclude=np.number).columns.tolist()

# Define the preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(sparse=False, handle_unknown='ignore'), non_numeric_cols),
        ('num', SimpleImputer(strategy='mean'), numeric_cols)
    ],
    remainder='passthrough'
)

# Define the GradientBoostingRegressor model
step2 = GradientBoostingRegressor(n_estimators=500, random_state=42)

# Define the pipeline
pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', step2)
])

# Fit the pipeline
pipe.fit(X_train, y_train)

# Predictions
y_pred = pipe.predict(X_test)

# Evaluate your model
print('R2 score:', r2_score(y_test, y_pred))
print('MAE:', mean_absolute_error(y_test, y_pred))

"""#XG BOOST"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify numeric and non-numeric columns
numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
non_numeric_cols = X_train.select_dtypes(exclude=np.number).columns.tolist()

# Define the preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), numeric_cols),
        ('cat', OneHotEncoder(sparse=False, handle_unknown='ignore'), non_numeric_cols)
    ],
    remainder='passthrough'
)

# Define the XGBRegressor model
step2 = XGBRegressor(n_estimators=45, max_depth=5, learning_rate=0.5)

# Define the pipeline
pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', step2)
])

# Fit the pipeline
pipe.fit(X_train, y_train)

# Predictions
y_pred = pipe.predict(X_test)

# Evaluate your model
print('R2 score:', r2_score(y_test, y_pred))
print('MAE:', mean_absolute_error(y_test, y_pred))

"""#Voting regressor"""

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, VotingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

# Example data
# Assuming X_train and X_test are pandas DataFrames
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Identify numeric and non-numeric columns
numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
non_numeric_cols = X_train.select_dtypes(exclude=np.number).columns.tolist()

# Define the preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(sparse=False, handle_unknown='ignore'), non_numeric_cols),
        ('num', SimpleImputer(strategy='mean'), numeric_cols)
    ],
    remainder='passthrough'
)

# Define the base models
rf = RandomForestRegressor(n_estimators=350, random_state=3, max_samples=None, max_features=0.75, max_depth=15)
gbdt = GradientBoostingRegressor(n_estimators=100, max_features=0.5)
xgb = XGBRegressor(n_estimators=25, learning_rate=0.3, max_depth=5)
et = ExtraTreesRegressor(n_estimators=100, random_state=3, max_samples=None, max_features=0.75, max_depth=10)

# Define the VotingRegressor with the base models
step2 = VotingRegressor([('rf', rf), ('gbdt', gbdt), ('xgb', xgb), ('et', et)], weights=[5, 1, 1, 1])

# Define the pipeline
pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('step2', step2)
])

# Fit the pipeline
pipe.fit(X_train, y_train)

# Predictions
y_pred = pipe.predict(X_test)

# Evaluate your model
print('R2 score:', r2_score(y_test, y_pred))
print('MAE:', mean_absolute_error(y_test, y_pred))

"""Conclusion--Gradient boost is giving best accuracy for the dataset"""











