
# coding: utf-8

# In[47]:

#!/usr/bin/env python


# In[1]:


import pandas as pd


# In[2]:


import numpy as np


# In[3]:


dtype_dict = {'bathrooms':float, 'waterfront':int, 'sqft_above':int, 'sqft_living15':float, 'grade':int, 'yr_renovated':int, 'price':float, 'bedrooms':float, 'zipcode':str, 'long':float, 'sqft_lot15':float, 'sqft_living':float, 'floors':str, 'condition':int, 'lat':float, 'date':str, 'sqft_basement':int, 'yr_built':int, 'id':str, 'sqft_lot':int, 'view':int}


# In[6]:


data_all=pd.read_csv('C:\csvFiles\kc_house_data.csv')


# In[7]:


data_test=pd.read_csv('C:\csvFiles\kc_house_test_data.csv')


# In[8]:


data_train=pd.read_csv('C:\csvFiles\kc_house_train_data.csv')


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




# In[24]:


get_numpy_data(data_all, features, output)


# In[25]:


def predict_outcome(feature_matrix, weights):
    weights=np.array(weights)
    perdictions= np.dot(feature_matrix, weights)
    return predictions


# In[26]:


def errors(output,predictions):
    errors=predictions-output
    return errors
    


# In[27]:


def features_derivative(errors, feature):
    derivative=np.dot(2,np.dot(features,errors))
    return derivative




# In[50]:

def regression_gradient_descent(feature_matrix, output, initial_weights, step_size, tolerance):
    converged = False
    #Initital weights are converted to numpy array
    weights = np.array([-100000., 1., 1.])
    while not converged:
        # compute the predictions based on feature_matrix and weights:
        predictions=predict_outcome(feature_matrix,weights)
        # compute the errors as predictions - output:
        error=errors('price',predictions)
        gradient_sum_squares = 0 # initialize the gradient
        # while not converged, update each weight individually:
        for i in range(len(weights)):
            # Recall that feature_matrix[:, i] is the feature column associated with weights[i]
            feature=feature_matrix[:, i]
            # compute the derivative for weight[i]:
            #predict=predict_outcome(feature,weights[i])
            #err=errors(output,predict)
            deriv=feature_derivative(error,feature)
            # add the squared derivative to the gradient magnitude
            gradient_sum_squares=gradient_sum_squares+(deriv**2)
            # update the weight based on step size and derivative:
            weights[i]=weights[i] - np.dot(7e-12,deriv)

        gradient_magnitude = math.sqrt(gradient_sum_squares)
        stdout.write("\r%d" % int(gradient_magnitude))
        stdout.flush()
        if gradient_magnitude < tolerance:
            converged = True
    return(weights)
#Example of Implementation
#Importing Training and Testing Data
# train_data=pd.read_csv("kc_house_train_data.csv")
# test_data=pd.read_csv("kc_house_test_data.csv")

# simple_features = ['sqft_living', 'sqft_living15']
# my_output= 'price'
# (simple_feature_matrix, output) = get_numpy_data(train_data, simple_features, my_output)
# initial_weights = np.array([-100000., 1., 1.])
# step_size = 7e-12
# tolerance = 2.5e7
# simple_weights = regression_gradient_descent(simple_feature_matrix, output,initial_weights, step_size,tolerance)
# print simple_weights





# In[56]:

get_numpy_data(data_all, ['sqft_living', 'sqft_living15'], 'price')


# In[60]:

(feature_matrix, output)= get_numpy_data(data_all,['sqft_living', 'sqft_living15'], 'price')


# In[62]:

regression_gradient_descent(feature_matrix, output, [-100000., 1., 1.], 7e-12, 2.5e7)


# In[ ]:



