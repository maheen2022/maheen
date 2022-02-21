#!/usr/bin/env python
# coding: utf-8

# # Data Analysis with Python
# House Sales in King County, USA
# This dataset contains house sale prices for King County, which includes Seattle. It includes homes sold between May 2014 and May 2015.
# 
# id :a notation for a house
# 
# date: Date house was sold
# 
# price: Price is prediction target
# 
# bedrooms: Number of Bedrooms/House
# 
# bathrooms: Number of bathrooms/bedrooms
# 
# sqft_living: square footage of the home
# 
# sqft_lot: square footage of the lot
# 
# floors :Total floors (levels) in house
# 
# waterfront :House which has a view to a waterfront
# 
# view: Has been viewed
# 
# condition :How good the condition is Overall
# 
# grade: overall grade given to the housing unit, based on King County grading system
# 
# sqft_above :square footage of house apart from basement
# 
# sqft_basement: square footage of the basement
# 
# yr_built :Built Year
# 
# yr_renovated :Year when house was renovated
# 
# zipcode:zip code
# 
# lat: Latitude coordinate
# 
# long: Longitude coordinate
# 
# sqft_living15 :Living room area in 2015(implies-- some renovations) This might or might not have affected the lotsize area
# 
# sqft_lot15 :lotSize area in 2015(implies-- some renovations)
# 
# You will require the following libraries

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
get_ipython().run_line_magic('matplotlib', 'inline')


# # 1.0 Importing the Data

# In[3]:


file_name='https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/coursera/project/kc_house_data_NaN.csv'
df=pd.read_csv(file_name)


# In[4]:


df.head()


# # Question 1

# Display the data types of each column using the attribute dtype, then take a screenshot and submit it, include your code in the image.

# In[6]:


print(df.dtypes)


# In[7]:


df.describe()


# # 2.0 Data Wrangling

# # Question 2

# In[8]:


df.drop(['id', 'Unnamed: 0'], axis=1, inplace=True)
df.describe()


# In[9]:


print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())


# In[12]:


mean=df['bedrooms'].mean()
df['bedrooms'].replace(np.nan,mean, inplace=True)


# In[11]:


mean=df['bathrooms'].mean()
df['bathrooms'].replace(np.nan,mean, inplace=True)


# In[13]:


print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())


# # 3.0 Exploratory data analysis

# # Question 3

# In[14]:


df['floors'].value_counts().to_frame()


# # Question 4

# In[15]:


sns.boxplot(x='waterfront', y='price', data=df)


# # Question 5

# In[16]:


sns.regplot(x='sqft_above', y='price', data=df)


# In[17]:


df.corr()['price'].sort_values()


# # Module 4: Model Development

# In[18]:


import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# In[19]:


X = df[['long']]
Y = df['price']
lm = LinearRegression()
lm
lm.fit(X,Y)
lm.score(X, Y)


# # Question 7

# In[20]:


features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]     


# In[21]:


X = df[features]
Y= df['price']
lm = LinearRegression()
lm.fit(X, Y)
lm.score(X, Y)


# # Question 8

# In[22]:


Input=[('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())]


# In[23]:


pipe=Pipeline(Input)
pipe


# In[24]:


pipe.fit(X,Y)


# In[25]:


pipe.score(X,Y)


# # Module 5: MODEL EVALUATION AND REFINEMENT

# In[26]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
print("done")


# In[27]:


features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]    
X = df[features ]
Y = df['price']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)


print("number of test samples :", x_test.shape[0])
print("number of training samples:",x_train.shape[0])


# # Question 9

# In[28]:


from sklearn.linear_model import Ridge


# In[29]:


RidgeModel = Ridge(alpha = 0.1)
RidgeModel.fit(x_train, y_train)
RidgeModel.score(x_test, y_test)


# # Question 10

# In[30]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
pr = PolynomialFeatures(degree=2)
x_train_pr = pr.fit_transform(x_train)
x_test_pr = pr.fit_transform(x_test)
poly = Ridge(alpha=0.1)
poly.fit(x_train_pr, y_train)
poly.score(x_test_pr, y_test)


# In[ ]:




