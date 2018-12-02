#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}


# In[6]:


data_all=pd.read_csv('C:\Users\N_Solgi\Desktop\Artificial Intelligence\machine learning\week04-assignment2\kc_house_data.csv')


# In[7]:


data_test=pd.read_csv('C:\Users\N_Solgi\Desktop\Artificial Intelligence\machine learning\week04-assignment2\kc_house_test_data.csv')


# In[8]:


data_train=pd.read_csv('C:\Users\N_Solgi\Desktop\Artificial Intelligence\machine learning\week04-assignment2\kc_house_train_data.csv')


# In[12]:


import math
from sys import stdout


# In[22]:


def get_numpy_data(data_all, features, output):
    data_all['constant']=1
    features=features+['constant']
    features_matrix=data_all['sqft_living'].as_matrix()
    output_array=np.array(data_all['price'])
    return (features_matrix, output_array)


# In[23]:


output=['price']
features=['sqft_living']


# In[24]:


get_numpy_data(data_all, features, output)


# In[25]:


def predict_outcome(features_matrix, weights):
    weights=np.array(weights)
    perdictions= np.dot(features_matrix, weights)
    return predictions


# In[26]:


def errors(output,predictions):
    errors=predictions-output
    return errors
    


# In[27]:


def features_derivative(errors, feature):
    derivative=np.dot(2,np.dot(features,errors))
    return derivative


# In[ ]:


cost(w)=SUM[(prediction - output)^2]
+l2 penalty*(w[0]^2 + w[1]^2 + ... +w[k]^2)

