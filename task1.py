#!/usr/bin/env python
# coding: utf-8

# In[1]:


#TASK1 Dev Kotak
import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  


# In[2]:


url = "http://bit.ly/w-data"
s_data = pd.read_csv(url)


# In[3]:


s_data.head(10)


# In[22]:


s_data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# In[23]:


X = s_data.iloc[:, :-1].values  
y = s_data.iloc[:, 1].values


# In[24]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                            test_size=0.2, random_state=0)


# In[25]:


from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit(X_train, y_train) 


# In[26]:


line = regressor.coef_*X+regressor.intercept_


# In[27]:


plt.scatter(X, y)
plt.plot(X, line);
plt.show()


# In[10]:


print(X_test) 
y_pred = regressor.predict(X_test) 
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df


# In[28]:


hours = [9.25]
own_pred = regressor.predict([hours])
print("No of Hours = {}".format(own_pred))


# In[29]:


from sklearn import metrics  
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred)) 


# In[42]:


#comparing the predicted marks with actual marks
plt.scatter(x=X_test, y=y_test , color='black')
plt.plot(X_test,y_pred ,color='red')
plt.title('Actual vs Predicted', size=20)
plt.ylabel('Percentage', size=12)
plt.xlabel('hours', size=12)
plt.show()


# In[ ]:


#we can say that if a student studies for 9.25 hours per day he is expected to score 93.6917 marks

