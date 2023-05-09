#!/usr/bin/env python
# coding: utf-8

# In[63]:


#Data injection 
import numpy as np                                                        # For pre-preocessing data
import pandas as pd                                                       # For pre-preocessing data
import matplotlib.pyplot as plt                                           # For visualisation
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns                                                     # For visualisation

                      # For training our Logistic Regression model
import scipy.stats as stats                                               # For training our model using Statsmodels
import statsmodels.api as sm                                              # For training our model using Statsmodels
from sklearn.metrics import classification_report,confusion_matrix        # For Performance metrics 
from sklearn.metrics import ConfusionMatrixDisplay                        # For plotting confusion matrix
#from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import cross_validate                        # For cross validation scores
from sklearn.model_selection import cross_val_score                       # For cross validation scores
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
                                                                          # For Performance metrics 
from statsmodels.stats.outliers_influence import variance_inflation_factor 
                                                                          # For Feature Selection
from sklearn.metrics import roc_auc_score                                 # For ROC AUC 
from sklearn.metrics import roc_curve                                     # For plotting ROC 
from sklearn.metrics import precision_recall_curve                        # For plotting Precision and Recall 

import os                                                                 # For changing home directory
from sklearn.model_selection import train_test_split                      # For train test split


pd.set_option('display.max_rows', 250)                                    # to show upto 250 rows in output
pd.set_option('display.max_colwidth',250)                                 # to show upto 250 cols in output
pd.set_option('display.float_format', lambda x: '%.5f' % x)               # customised format for pandas dataframe output


import warnings
warnings.filterwarnings('ignore')    
# To supress warnings


import os                                        # for customising home directory
import math                
import numpy as np
import pandas as pd
import seaborn as sns                            # for plots                  


from statsmodels.formula import api                # library used for model training ( better statisics)
from sklearn.linear_model import LinearRegression  # Another library used for model training 
from sklearn.feature_selection import RFE          # library used to reduce collinearity and feature selection
from sklearn.preprocessing import StandardScaler   # used for Standardasing
from sklearn.model_selection import train_test_split # used for train/test splits

from IPython.display import display # function used to render appropriate mehod to display objects # new function in week


from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error # used for performance metrics

import matplotlib.pyplot as plt # used for plotting
import warnings # used to set how much warnings should be displayed
warnings.filterwarnings('ignore')


plt.style.use('ggplot')     
titanic = pd.read_csv('D:/DS/resume projects/ML Titanic/Titanic-Dataset.csv')


# In[64]:


titanic


# In[65]:


titanic.nunique()


# In[66]:


#data cleaning and feature extraction 
titanic['FamilySize'] = titanic['SibSp'] + titanic['Parch'] + 1
titanic['Embarked'] = titanic['Embarked'].fillna(titanic['Embarked'].mode()[0])
titanic['Fare'] = titanic['Fare'].fillna(titanic['Fare'].median())
titanic['CategoricalFare'] = pd.qcut(titanic['Fare'], 3)



age_avg = titanic['Age'].mean()
age_std = titanic['Age'].std()
age_null_count = titanic['Age'].isnull().sum()
age_null_random_list = np.random.randint(age_avg - 3*age_std, age_avg + 3*age_std, size=age_null_count)
titanic.loc[titanic['Age'].isna(), 'Age'] = age_null_random_list
titanic['Age'] = titanic['Age'].astype(int)
titanic['CategoricalAge'] = pd.cut(titanic['Age'], 5)


# In[67]:


#label encoding

titanic['Sex'] = titanic['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
    
titanic['Embarked'] = titanic['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

titanic.loc[ titanic['Fare'] <= 7.91, 'Fare']                               = 0
titanic.loc[(titanic['Fare'] > 7.91) & (titanic['Fare'] <= 14.454), 'Fare'] = 1
titanic.loc[(titanic['Fare'] > 14.454) & (titanic['Fare'] <= 31), 'Fare']   = 2
titanic.loc[ titanic['Fare'] > 31, 'Fare']                                  = 3
titanic['Fare'] = titanic['Fare'].astype(int)

titanic.loc[ titanic['Age'] <= 16, 'Age']                          = 0
titanic.loc[(titanic['Age'] > 16) & (titanic['Age'] <= 32), 'Age'] = 1
titanic.loc[(titanic['Age'] > 32) & (titanic['Age'] <= 48), 'Age'] = 2
titanic.loc[(titanic['Age'] > 48) & (titanic['Age'] <= 64), 'Age'] = 3
titanic.loc[ titanic['Age'] > 64, 'Age']   


# In[68]:


df=titanic.drop(['Name','Ticket','Cabin','PassengerId'],axis=1)
df


# In[69]:


catcol=['Pclass','Sex','SibSp','Parch','Embarked','FamilySize','CategoricalFare','CategoricalAge']
dfht=pd.DataFrame({})
for i in catcol:
    dfht=pd.concat([dfht,pd.get_dummies(df[i],drop_first=True)],axis=1,)
dfht


# In[70]:


from sklearn.model_selection import train_test_split
x=dfht#.drop(['Survived'],axis=1)
y=df['Survived']
xn,xs,yn,ys=train_test_split(x,y,train_size=0.8)


# In[71]:


#rfe
ln=[]
ls=[]
from sklearn.linear_model import LinearRegression

col=len(x.columns)
for i in range(col):
    lr=LinearRegression()
    rfe=RFE(lr,n_features_to_select=xn.shape[1]-i)
    rfe=rfe.fit(xn,yn)
    
    LR=LinearRegression()
    cn=xn.loc[:,rfe.support_]
    cs=xs.loc[:,rfe.support_]
    LR.fit(cn,yn)
    ypn=LR.predict(cn)
    yps=LR.predict(cs)
    
    ls.append(r2_score(ys,yps))
    ln.append(r2_score(yn,ypn))

plt.plot(ls,label='Test')
plt.plot(ln,label='Train')
plt.title('R2 curve')
plt.legend()
plt.grid()
plt.show()


# In[73]:


lr=LinearRegression()
rfe=RFE(lr,n_features_to_select=xn.shape[1])
rfe=rfe.fit(xn,yn)

LR=LinearRegression()
cn=xn.loc[:,rfe.support_]
cs=xs.loc[:,rfe.support_]
LR.fit(cn,yn)
ypn=LR.predict(cn)
yps=LR.predict(cs)

print(r2_score(ys,yps))
print(r2_score(yn,ypn))


# In[74]:


from sklearn.linear_model import LogisticRegression 
lr=LogisticRegression()
lr=lr.fit(xn,yn)
ypn=lr.predict(xn)
yps=lr.predict(xs)
ypnn=lr.predict_proba(xs)


# In[75]:


pd.concat([pd.DataFrame(ypnn),pd.DataFrame(yn)],axis=1)


# In[81]:


sen,sep,threshold1=roc_curve(ys,pd.DataFrame(ypnn)[1])
plt.plot(sen,sep,color='green')
plt.plot([0,1],[0,1],'--')
plt.title('ROC curve')
plt.ylabel('Sensitivity')
plt.xlabel('1-Specificity')
plt.show()


# In[76]:


cnns=confusion_matrix(ys,yps)
sns.heatmap(cnns,annot=True)
plt.show()
cnnn=confusion_matrix(yn,ypn)
sns.heatmap(cnnn,annot=True)


# In[77]:


accn=accuracy_score(yn,ypn)
accs=accuracy_score(ys,yps)
print('Train',str(accn*100)+'%')
print('test',str(accs*100)+'%')


# In[82]:


from sklearn.metrics import recall_score
gmts=classification_report(ys,yps)
gmtn=classification_report(yn,ypn)
print('Test:-\n',gmts,'\n','Train:-\n',gmtn)


# In[ ]:




