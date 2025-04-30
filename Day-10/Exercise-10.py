#!/usr/bin/env python
# coding: utf-8

# Train SVM classifier using sklearn digits dataset (i.e. from sklearn.datasets import load_digits) and then,
# 
# Measure accuracy of your model using different kernels such as rbf and linear.
# 
# Tune your model further using regularization and gamma parameters and try to come up with highest accurancy score
# 
# Use 80% of samples as training data size

# In[2]:


import pandas as pd
from sklearn.datasets import load_digits
digits = load_digits()


# In[3]:


dir(digits)


# In[4]:


digits.target


# In[5]:


digits.target_names


# In[6]:


digits.data


# In[7]:


digits.images


# In[8]:


len(digits.data)


# In[9]:


df = pd.DataFrame(digits.data, digits.target)
df.head()


# In[10]:


df['target'] = digits.target
df.head(20)


# In[38]:


from sklearn.model_selection import train_test_split


# In[39]:


X = df.drop(['target'],axis='columns')
y = df.target


# In[41]:


X_train, X_test , y_train, y_test = train_test_split(X,y,test_size=0.2)


# # Using RBF Kernel

# In[42]:


from sklearn.svm import SVC
rbf_model = SVC(kernel='rbf')


# In[43]:


len(X_train)


# In[44]:


len(X_test)


# In[45]:


rbf_model.fit(X_train, y_train)


# In[46]:


rbf_model.score(X_test,y_test)


# # Using Linear Kernel

# In[47]:


linear_model = SVC(kernel='linear')
linear_model.fit(X_train, y_train)


# In[48]:


linear_model.score(X_test, y_test)


# In[ ]:




