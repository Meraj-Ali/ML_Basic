#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()


# In[2]:


dir(iris)


# In[3]:


iris.feature_names


# In[4]:


iris.target_names


# In[5]:


len(iris.data)


# In[6]:


df = pd.DataFrame(iris.data, columns=iris.feature_names)
df.head()


# In[7]:


df['target'] = iris.target
df.head()


# In[8]:


iris.target


# In[9]:


df[df.target==1].head()


# In[10]:


df['flower_name'] = df.target.apply(lambda x: iris.target_names[x])
df.head()


# In[11]:


df[45:55]


# In[12]:


df0 = df[:50]
df1 = df[50:100]
df2 = df[100:]


# In[13]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[14]:


df.columns


# In[15]:


plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.scatter(df0['sepal length (cm)'], df0['sepal width (cm)'],color='green',marker='+')
plt.scatter(df1['sepal length (cm)'], df1[ 'sepal width (cm)'], color='blue', marker='*')


# In[16]:


plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.scatter(df0['petal length (cm)'], df0['petal width (cm)'],color='red', marker='+')
plt.scatter(df1['petal length (cm)'], df1['petal width (cm)'],color='black', marker='*')


# In[18]:


from sklearn.model_selection import train_test_split


# In[19]:


X = df.drop(['target', 'flower_name'], axis='columns')
y = df.target


# In[20]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


# In[21]:


len(X_train)


# In[22]:


len(X_test)


# In[23]:


from sklearn.svm import SVC
model = SVC()


# In[24]:


model.fit(X_train,y_train)


# In[25]:


model.score(X_test, y_test)


# In[26]:


X_test


# In[27]:


model.predict([[5.5, 2.5, 4.0, 1.3]])


# In[28]:


# Regularization
model_C=SVC(C=10)
model_C.fit(X_train, y_train)


# In[29]:


model_C.score(X_test, y_test)


# In[30]:


# Gamma
model_g = SVC(gamma=5)
model_g.fit(X_train, y_train)


# In[31]:


model_g.score(X_test, y_test)


# In[33]:


# Kernal
model_linear_kernal=SVC(kernel='linear')
model_linear_kernal.fit(X_train, y_train)


# In[35]:


model_linear_kernal.score(X_test, y_test)


# In[ ]:




