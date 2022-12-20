#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture

import seaborn as sns


# In[2]:


df = pd.read_csv('nutrition_values.csv',sep= ";")


# In[3]:


df.head()


# In[4]:


df.describe()


# In[5]:


df.isnull().sum()


# In[6]:


df.info()


# We can see that three columns , 6,7,8 are objects insted of integers or foalts giving a clear indication that they. have inproperly stuctured data that needs to be cleared . Also we understand the columns 0,1,2,3 need to be numberalised for prediction sake . 

# ## PreProcessing Data 
# Lets take a deep dive into the data to clean it better 

# In[7]:


print(df["Total Fat (g)"].unique())


# We can see few numbers has ',' insted of decimal points 

# In[8]:


df["Total Fat (g)"] = df["Total Fat (g)"].str.replace(',','.')


# In[9]:


df["Total Fat (g)"].unique()


# In[10]:


print(df["Saturated Fat (g)"].unique())
print(df["Trans Fat (g)"].unique())


# the data in these are distributed as strings rather than integers for the above reason 

# In[11]:


df["Saturated Fat (g)"] = df["Saturated Fat (g)"].str.replace(',','.')


# In[12]:


df["Saturated Fat (g)"] = df["Saturated Fat (g)"].str.replace(',','.')


# In[13]:


df["Total Fat (g)"] = df["Total Fat (g)"].str.replace(' -   ','0')


# In[14]:


df["Saturated Fat (g)"] = df["Saturated Fat (g)"].str.replace(' -   ','0')


# In[15]:


df["Trans Fat (g)"] = df["Trans Fat (g)"].str.replace(' -   ','0')


# In[16]:


df["Trans Fat (g)"] = df["Trans Fat (g)"].str.replace(',','.')


# In[17]:


print(df["Trans Fat (g)"].unique());
print(df["Saturated Fat (g)"].unique());
print(df["Trans Fat (g)"].unique());


# In[18]:


df[["Total Fat (g)", "Saturated Fat (g)", "Trans Fat (g)"]] = df[["Total Fat (g)", "Saturated Fat (g)", "Trans Fat (g)"]].astype(float)


# In[19]:


df.info()


# In[20]:


df["Chain"].unique()


# In[21]:


for i in range (len(df.index)):
          
    if df["Chain"][i] == 'Burger King':
            df["Chain"][i] = '1'
    if df["Chain"][i] == 'Mc Donalds':
            df["Chain"][i] = '2'
          
          


# In[22]:


df["Serving Size (g)"].unique()


# In[23]:


df["Serving Size (g)"] = df["Serving Size (g)"].fillna(0)


# In[24]:


df["Serving Size (g)"] = df["Serving Size (g)"].str.replace(',','.')


# In[25]:


df[["Chain", "Serving Size (g)"]] = df[["Chain", "Serving Size (g)"]].astype(float)


# In[26]:


df.info()


# In[27]:


df.drop(['Item','Type'],axis = 1 , inplace = True)


# We can see that the item and type values are varied and cant be changed to numericals so can be used for categorical values 

# In[28]:


corr = df.corr()
plt.figure(figsize=(10, 10))
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns,
        cmap="Blues", annot=True)


# From the above data we can see that Chain feature is dominantly independent , where as calories , fat, sodium and protine are interdependent We can try and make a matrix model and a k means model of these features 

# In[29]:


fea = ['Total Fat (g)','Protein (g)']


# In[30]:


X = df[fea]


# In[31]:


z = StandardScaler()


# In[32]:


X[fea] = z.fit_transform(X)


# In[33]:


mod = GaussianMixture(n_components = 3)
mod.fit(X)
pred = mod.predict(X)


# In[34]:


clus = mod.predict_proba(X)
fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(clus)
 
# Show plot
plt.show()


# In[35]:


plt.figure(figsize=(10, 10))
plt.scatter(X['Total Fat (g)'], X['Protein (g)'], c =clus);


# In[ ]:




