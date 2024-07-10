#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
import warnings
import os


# In[2]:


df1  = pd.read_excel(r"C:\Users\COMPUTER CARE\Documents\campusx\Credit Risk Modelling Using ML\case_study1.xlsx")


# In[3]:


df1.head()   # -99999 in last two columns are null data


# In[4]:


df1.info()


# In[5]:


df1.describe()


# In[6]:


# in data we saw only 40 rows are with (-9999) out of 51K, we can drop these, 
df1 = df1.loc[df1['Age_Oldest_TL'] != -99999]


# In[ ]:





# In[7]:


df2 = pd.read_excel(r"C:\Users\COMPUTER CARE\Documents\campusx\Credit Risk Modelling Using ML\case_study2.xlsx")


# In[8]:


df2.head()


# In[9]:


df2.info()


# In[10]:


df2.describe()


# In[11]:


# IN df2 many columns have significant Null Values, if Column having null values more than 10K,we are removing particualr Column and if less than 10K, we will remove those rows
columns_to_be_removed = []
# dg.columns creates  --> list of columns

for i in df2.columns:
    if df2.loc[df2[i] == -99999].shape[0] > 10000:
        columns_to_be_removed .append(i)


# In[12]:


df2 = df2.drop(columns_to_be_removed, axis =1)


for i in df2.columns:
    df2 = df2.loc[ df2[i] != -99999 ]


# In[13]:


# Checking common column names
for i in list(df1.columns):
    if i in list(df2.columns):
        print (i)


# In[14]:


print(df1.shape)


# In[15]:


print(df2.shape)  # (We lost around 9K rows and 8 colums)


# In[16]:


# Merge the two dataframes, inner join so that no nulls are present
df = pd. merge ( df1, df2, how ='inner', left_on = ['PROSPECTID'], right_on = ['PROSPECTID'] )


# In[17]:


df.shape


# In[18]:


df.isna().sum()


# In[19]:


df.head()


# In[20]:


df.info()


# In[21]:


#we will divide features into Numerical and Catergorical 

# check how many columns are categorical
for i in df.columns:
    if df[i].dtype == 'object':
        print(i)


# In[22]:


df['MARITALSTATUS'].value_counts()
df['first_prod_enq2'].value_counts()
# Approved Flag is Target Variable


# In[23]:


#WE want to check if these categorical features are associated to target variable
# Let's say maritalstatus and approved flag


# we cannot marritalstatus to label encoding --> when data is ordinal in nature(like we can merit the category(eg high medium low))


# In[24]:


# Chi-square test
for i in ['MARITALSTATUS', 'EDUCATION', 'GENDER', 'last_prod_enq2', 'first_prod_enq2']:
    chi2, pval, _, _ = chi2_contingency(pd.crosstab(df[i], df['Approved_Flag']))
    print(i, '---', pval)


# In[25]:


# Since all the categorical values have p-value <0.05, rejecting H0 (null hypothesis) , so we are going to accept all the cateogorical columns


# VIF for numerical columns
numeric_columns = []
for i in df.columns:
    if df[i].dtype != 'object' and i not in ['PROSPECTID','Approved_Flag']:
        numeric_columns.append(i)
        
        
        
print(len(numeric_columns))
# we'll have to check if these 72 columns are related btw themselves (multicollinearity,coorelation, VIF) 


# In[26]:


# VIF sequentially check

vif_data = df[numeric_columns]
total_columns = vif_data.shape[1]
columns_to_be_kept = []
column_index = 0



for i in range (0,total_columns):
    
    vif_value = variance_inflation_factor(vif_data, column_index)
    print (column_index,'---',vif_value)
    
    
    if vif_value <= 6:   # 6 is the threshold value we are keeping on hold to check for multicollinearity
        columns_to_be_kept.append( numeric_columns[i] )
        column_index = column_index+1
    
    else:
        vif_data = vif_data.drop([ numeric_columns[i] ] , axis=1)


# In[27]:


print(len(columns_to_be_kept))  # 39 , initially 72


# In[28]:


# We we use Anova test for this --> 39 cloumns compared with Target Variable

# check Anova for columns_to_be_kept 

from scipy.stats import f_oneway

columns_to_be_kept_numerical = []

for i in columns_to_be_kept:
    a = list(df[i])  
    b = list(df['Approved_Flag'])  
    
    group_P1 = [value for value, group in zip(a, b) if group == 'P1']
    group_P2 = [value for value, group in zip(a, b) if group == 'P2']
    group_P3 = [value for value, group in zip(a, b) if group == 'P3']
    group_P4 = [value for value, group in zip(a, b) if group == 'P4']


    f_statistic, p_value = f_oneway(group_P1, group_P2, group_P3, group_P4)

    if p_value <= 0.05:
        columns_to_be_kept_numerical.append(i)


# In[29]:


print(len(columns_to_be_kept_numerical))


# In[30]:


# listing all the final features
features = columns_to_be_kept_numerical + ['MARITALSTATUS', 'EDUCATION', 'GENDER', 'last_prod_enq2', 'first_prod_enq2']
df = df[features + ['Approved_Flag']]


# In[31]:


# Label encoding for the categorical features
['MARITALSTATUS', 'EDUCATION', 'GENDER' , 'last_prod_enq2' ,'first_prod_enq2']



print(df['MARITALSTATUS'].unique()) 
print(df['EDUCATION'].unique())
print(df['GENDER'].unique())
print(df['last_prod_enq2'].unique())
print(df['first_prod_enq2'].unique())

# for EDUCATION we are doing Label Encoding and for other we are doing one Hot Encoding


# In[32]:


# Ordinal feature -- EDUCATION
# SSC            : 1
# 12TH           : 2
# GRADUATE       : 3
# UNDER GRADUATE : 3
# POST-GRADUATE  : 4
# OTHERS         : 1
# PROFESSIONAL   : 3


# Others has to be verified by the business end user 




df.loc[df['EDUCATION'] == 'SSC',['EDUCATION']]              = 1
df.loc[df['EDUCATION'] == '12TH',['EDUCATION']]             = 2
df.loc[df['EDUCATION'] == 'GRADUATE',['EDUCATION']]         = 3
df.loc[df['EDUCATION'] == 'UNDER GRADUATE',['EDUCATION']]   = 3
df.loc[df['EDUCATION'] == 'POST-GRADUATE',['EDUCATION']]    = 4
df.loc[df['EDUCATION'] == 'OTHERS',['EDUCATION']]           = 1
df.loc[df['EDUCATION'] == 'PROFESSIONAL',['EDUCATION']]     = 3


# In[33]:


df['EDUCATION'].value_counts()


# In[34]:


df['EDUCATION'] = df['EDUCATION'].astype(int)
df.info()


# In[35]:


# One-Hot Encoding
df_encoded = pd.get_dummies(df, columns=['MARITALSTATUS','GENDER', 'last_prod_enq2' ,'first_prod_enq2'])


# In[36]:


df.head()


# In[37]:


df_encoded.info()


# In[38]:


k = df_encoded.describe()


# In[39]:


# Machine Learning Model Fitting

# Data processing

# 1. Random Forest

y = df_encoded['Approved_Flag']
x = df_encoded. drop ( ['Approved_Flag'], axis = 1 )



# In[40]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[41]:


rf_classifier = RandomForestClassifier(n_estimators = 200, random_state=42)


# In[42]:


rf_classifier.fit(x_train, y_train)
y_pred = rf_classifier.predict(x_test)


# In[43]:


accuracy = accuracy_score(y_test, y_pred)
print ()
print(f'Accuracy: {accuracy}')
print ()
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)


# In[44]:


for i, v in enumerate(['p1', 'p2', 'p3', 'p4']):
    print(f"Class {v}:")
    print(f"Precision: {precision[i]}")
    print(f"Recall: {recall[i]}")
    print(f"F1 Score: {f1_score[i]}")
    print()


# In[45]:


# 2. xgboost

import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

xgb_classifier = xgb.XGBClassifier(objective='multi:softmax',  num_class=4)



y = df_encoded['Approved_Flag']
x = df_encoded. drop ( ['Approved_Flag'], axis = 1 )


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)




# In[46]:


x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=42)


# In[47]:


xgb_classifier.fit(x_train, y_train)


# In[48]:


y_pred = xgb_classifier.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print ()
print(f'Accuracy: {accuracy:.2f}')
print ()


# In[49]:


precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)

for i, v in enumerate(['p1', 'p2', 'p3', 'p4']):
    print(f"Class {v}:")
    print(f"Precision: {precision[i]}")
    print(f"Recall: {recall[i]}")
    print(f"F1 Score: {f1_score[i]}")
    print()


# In[50]:


# 3. Decision Tree
from sklearn.tree import DecisionTreeClassifier


y = df_encoded['Approved_Flag']
x = df_encoded. drop ( ['Approved_Flag'], axis = 1 )




# In[51]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)



# In[52]:


dt_model = DecisionTreeClassifier(max_depth=20, min_samples_split=10)
dt_model.fit(x_train, y_train)


# In[53]:


y_pred = dt_model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print ()
print(f"Accuracy: {accuracy:.2f}")
print ()



# In[54]:


precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)

for i, v in enumerate(['p1', 'p2', 'p3', 'p4']):
    print(f"Class {v}:")
    print(f"Precision: {precision[i]}")
    print(f"Recall: {recall[i]}")
    print(f"F1 Score: {f1_score[i]}")
    print()


# In[55]:


# xgboost is giving me best results
# We will further finetune it


# Apply standard scaler 

from sklearn.preprocessing import StandardScaler

columns_to_be_scaled = ['Age_Oldest_TL','Age_Newest_TL','time_since_recent_payment',
'max_recent_level_of_deliq','recent_level_of_deliq',
'time_since_recent_enq','NETMONTHLYINCOME','Time_With_Curr_Empr']

for i in columns_to_be_scaled:
    column_data = df_encoded[i].values.reshape(-1, 1)
    scaler = StandardScaler()
    scaled_column = scaler.fit_transform(column_data)
    df_encoded[i] = scaled_column


# In[56]:


import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

xgb_classifier = xgb.XGBClassifier(objective='multi:softmax',  num_class=4)



y = df_encoded['Approved_Flag']
x = df_encoded. drop ( ['Approved_Flag'], axis = 1 )


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=42)






# In[57]:


xgb_classifier.fit(x_train, y_train)
y_pred = xgb_classifier.predict(x_test)



# In[58]:


accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')




# In[59]:


precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)

for i, v in enumerate(['p1', 'p2', 'p3', 'p4']):
    print(f"Class {v}:")
    print(f"Precision: {precision[i]}")
    print(f"Recall: {recall[i]}")
    print(f"F1 Score: {f1_score[i]}")
    print()


# In[60]:


# No improvement in metrices
# Hyperparameter tuning in xgboost

from sklearn.model_selection import GridSearchCV
x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=42)

# Define the XGBClassifier with the initial set of hyperparameters
xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=4)



# In[61]:


# Define the parameter grid for hyperparameter tuning

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
}

grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(x_train, y_train)



# In[62]:


# Print the best hyperparameters
print("Best Hyperparameters:", grid_search.best_params_)

# Evaluate the model with the best hyperparameters on the test set
best_model = grid_search.best_estimator_
accuracy = best_model.score(x_test, y_test)
print("Test Accuracy:", accuracy)



# Best Hyperparameters: {'learning_rate': 0.2, 'max_depth': 3, 'n_estimators': 200}


# Based on risk appetite of the bank, you will suggest P1,P2,P3,P4 to the business end user


# In[ ]:




