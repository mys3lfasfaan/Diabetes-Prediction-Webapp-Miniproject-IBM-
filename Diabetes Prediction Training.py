#!/usr/bin/env python
# coding: utf-8

# In[51]:


import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

import seaborn as sns
sns.set_style("darkgrid")


# In[52]:


path = "C:/Users/Abrar sharif/OneDrive/Desktop/Asfaan Mini/diabetes_prediction_dataset.csv"
dataset = pd.read_csv(path)
dataset.head()


# # EDA

# In[53]:


dataset.describe()


# In[54]:


dataset.dtypes


# In[55]:


dataset['age'] = dataset['age'].astype(int)
dataset['blood_glucose_level'] =  dataset['blood_glucose_level'].astype(float)


# In[56]:


dataset['gender'].value_counts()


# In[57]:


dataset['smoking_history'].value_counts()


# In[58]:


dataset = dataset.drop(columns = 'smoking_history')


# In[59]:


sns.catplot(data = dataset, x = 'diabetes', y = 'age', kind = 'box', hue = 'gender')
plt.show()


# In[60]:


sns.relplot(data = dataset, x = 'HbA1c_level', y =  'blood_glucose_level', kind = 'line', hue = 'gender', ci = None, markers = True)
plt.show()


# In[61]:


# As trans sample is very less... we are ignoring it


# In[62]:


trans = dataset.loc[dataset['gender'] == "Other"]
trans


# In[63]:


trans.index


# In[64]:


d2 = dataset.drop(index = trans.index, axis = 'index')
d2


# In[65]:


d2['gender'].value_counts()


# # Preprocessing 

# In[66]:


X = d2.iloc[:,[1,2,3,4,5,6]]
y = d2.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# # Using Logistic Regression

# In[67]:


results = []

clf = LogisticRegression()

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

results = [
    accuracy_score(y_test, y_pred),
    precision_score(y_test, y_pred, average='weighted'),
    recall_score(y_test, y_pred, average='weighted'),
    f1_score(y_test, y_pred, average='weighted')
    ]

cm = confusion_matrix(y_pred, y_test)
print("Accuracy score: ", accuracy_score(y_pred,y_test)*100)
print("\n\n\n", cm, "\n\n\n")
print(results)


# In[68]:


ypredclass = clf.predict_proba(X_test)
ypredclass


# In[69]:


# Get the maximum probability
max_prob = np.max(ypredclass, axis=1)

# Get the index of the class with the maximum probability
max_index = np.argmax(ypredclass, axis=1)

# Round the maximum probability to 3 decimal places
max_prob = np.round(max_prob, 3)

# Make DataFrame
prob_matrix = pd.DataFrame({
    'class_index': max_index,
    'probability': max_prob
})

# Print the maximum probability
prob_matrix


# In[70]:


prob_matrix.loc[prob_matrix['class_index'] == 1]


# # Save the model

# In[73]:


import pickle

with open('C:/Users/Abrar sharif/OneDrive/Desktop/Asfaan Mini/finalized_model.sav', 'wb') as f:
    pickle.dump(clf, f)


# In[ ]:





# # Prediction with GUI

# In[72]:


### In desktop as web app

