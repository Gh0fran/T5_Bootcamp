#!/usr/bin/env python
# coding: utf-8

# # Passenger Satisfaction - Classification Problem 

# Building a customer-centric airline means offering passengers memorable and lasting experiences. Each customer has their own set of expectations when it comes to their interactions with airlines. Airlines have the opportunity to provide a personalized travel experience that matches or exceeds those specific expectations.
# This project aims to help any airline to become a Customer-Centric Airline by predicting which flight services affect most on customer satisfaction based on passenger characteristics.
# 

# There are 3 types of services that airlines offer for passengers: 
# 1-Onboard Services = Pre-Flight services  and  Post-Flight services 
# 2-In-Flight services 

# In[178]:


from IPython.display import Image 
Image(url= "https://github.com/Gh0fran/T5_Bootcamp/blob/main/Data%20Description.JPG?raw=true")


# #  Q: Which service should airlines focus on that increase customer satisfaction?  

# # Part 1: Exploratory Data Analysis

# ## Importing Libraries

# In[179]:


# importing libraries
import sys 
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge 
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix ,accuracy_score , plot_confusion_matrix ,plot_roc_curve

import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'png'")

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Dataset

# In[180]:


# read data
data = pd.read_csv('US_Airline_passenger_satisfaction_survey.csv')
Original_data= data


# In[181]:


data.head(5)


# ## Size of the dataset 

# In[182]:


data.shape


# ## Identification of variables and data types

# In[183]:


data.info()


# ### Numerical variables:
# #### Disceret
# Age, Flight Distance ,Departure Delay in Minutes 
# ####  Satisfaction ratings on a scale from 1 to 5 
# 1:Very unsatisfied , 2:unsatisfied , 3:Neutral , 4:satisfied , 5:Very satisfied
# 
# Inflight wifi service , Departure/Arrival time convenient, Ease of Online booking,Gate location, Food and drink, Online boarding, Seat comfort, Inflight entertainment, On-board service, Leg room service,Baggage handling, Checkin service, Inflight service, Cleanliness
# 
# #### Continuous
# Arrival Delay in Minutes
# 
# ### Categorical variables:
# satisfaction_v2, Gender, Customer Type, Type of Travel, Class
# 
# 

# The target value is satisfaction_v2, and the rest 23 features can be assumed as the predictor variables.

# ## Columns Name Modification

# In[184]:


# modifying features name to remove white spaces and /
data.columns = data.columns.str.replace(' ', '_')
data.columns = data.columns.str.replace('/', '_')
data.columns = data.columns.str.replace('-', '_')
data.columns = data.columns.str.replace('satisfaction_v2', 'satisfaction')


# ## Finding null values

# In[185]:


data.apply(lambda x: sum(x.isnull()),axis=0)


# In[186]:


# here we can find missing 393 vaules for 'Arrival Delay in Minutes' column
# it equals to 0.3 % of total responses , we can drop them. 
data.dropna(subset=['Arrival_Delay_in_Minutes'], inplace=True)


# In[187]:


data.shape


# ## Statistical Summary of Numeric Variables:

# In[188]:


data.describe()


# ### Findings
# Scale features should be from 1 to 5, having 0 as min indicates un-answered question 

# Removing rows with 0 rating 

# In[189]:


# skipping the score of Departure_Arrival_time_convenient 
Keep_answered_mask = ((data.Departure_Arrival_time_convenient  != 0) &(data.Inflight_wifi_service  != 0)
                     & (data.Ease_of_Online_booking  != 0) &(data.Gate_location  != 0) &(data.Food_and_drink  != 0)
                     & (data.Online_boarding  != 0) & (data.Leg_room_service  != 0) & (data.Checkin_service  != 0)
                     & (data.Inflight_service  != 0) & (data.Cleanliness  != 0) & (data.Baggage_handling  != 0))

data = data[Keep_answered_mask]
data.shape


# # Categorical features visualization

# ## Data Balance

# In[190]:


sns.countplot(data.satisfaction , palette='Pastel1' )


# The reporting Satisfied 'satisfied' are balanced with passengers reporting Neutral/Dissatisfied . 
# The high number entries in negative class is not surprising since ‘Neutral/Dissatisfied’ does not necessarily means dissatisfaction

# In[191]:


gender = sns.catplot(x="Gender", data=data, kind='count', hue='satisfaction', palette='Pastel1' ,height=3.27, aspect=4/2)
gender.set_ylabels('Gender vs Passenger Satisfaction')


# In[192]:


g = sns.catplot(x="Age", data=data, aspect=3.0, kind='count', hue='satisfaction',palette='Pastel1', order=range(5, 85))
g.set_ylabels('Age vs Passenger Satisfaction')
g.set_xlabels('Age')


# In[193]:


g = sns.catplot(x="Customer_Type", data=data, kind='count', hue='satisfaction', palette='Pastel1',height=3.27, aspect=4/2  )
g.set_ylabels('Customer Type vs Passenger Satisfaction')


# In[194]:


g = sns.catplot(x="Type_of_Travel", data=data, kind='count', hue='satisfaction', palette='Pastel1',height=3.27, aspect=4/2 )
g.set_ylabels('Travel Type vs Passenger Satisfaction')


# In[195]:


gender = sns.catplot(x="Class", data=data,  kind='count', hue='satisfaction', palette='Pastel1' , height=3.27, aspect=4/2)
gender.set_ylabels('Class vs Passenger Satisfaction')


# When I further segment the satisfaction classes by Type_of_Travel, we see that Personal travel customers have a lower ratio of satisfaction. In addition, when I segment the satisfaction classes by class , it is observed that customers on Eco class have significantly lower ratio of satisfaction. 
# In both cases, higher expectation of experience might have played a part in the reduction of satisfaction.

# # Q: Do Class and Travel Type play a role in passenger satisfactions?

# # 

# In[196]:


fig, ax_rows = plt.subplots(7, 2, figsize=(12, 30))
features=['Inflight_wifi_service','Departure_Arrival_time_convenient','Ease_of_Online_booking','Gate_location','Food_and_drink','Online_boarding','Seat_comfort','Inflight_entertainment','On_board_service','Leg_room_service','Baggage_handling','Checkin_service','Inflight_service','Cleanliness']
for feature in features:
    feature_data = data.groupby([feature]).size().reset_index(name=feature+'counts')
    degree = features.index(feature)
    ax_row_left, ax_row_right = ax_rows[degree//2]
    if degree%2 == 0:
        ax = ax_row_left
    else:
        ax = ax_row_right
    sns.barplot(x=feature, y=feature+'counts', palette="Pastel1", data=feature_data ,ax=ax )


    


# ## Outliers  in Numerical

# In[197]:


#Return values at the given quantile over requested axis
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
print(IQR)


# In[198]:


data = data[~((data < (Q1 - 1.5 * IQR)) |(data > (Q3 + 1.5 * IQR))).any(axis=1)]
data.shape


# In[199]:


plot = data.boxplot(column='Flight_Distance')


# ## Label Encoding of Categorical Variables

# In[200]:


data['Male'] = data['Gender'].apply(lambda x: "1" if x == "Male" else "0")
data['Male'] = (data['Male'].astype(int))
data['Loyal'] = data['Customer_Type'].apply(lambda x: "1" if x == "Loyal Customer" else "0")
data['Loyal'] = (data['Loyal'].astype(int))
data['Business_Class'] = data['Class'].apply(lambda x: "1" if x == "Business" else "0")
data['Business_Class'] = (data['Business_Class'].astype(int))
data['Personal_Travel'] = data['Type_of_Travel'].apply(lambda x: "1" if x == "Business travel" else "0")
data['Personal_Travel'] = (data['Personal_Travel'].astype(int))


# In[201]:


data['satisfied'] = data['satisfaction'].apply(lambda x: "1" if x == "satisfied" else "0")
data['satisfied'] = (data['satisfied'].astype(int))


# In[202]:


data.drop(['Gender','Customer_Type', 'Class','Type_of_Travel','satisfaction'] , inplace =True ,axis=1)


# In[203]:


pd.set_option("display.max_columns", None)
data.corr()


# High correlations:
# "Inflight_wifi_service" and "Ease_of_Online_booking" corr=0.686577
# "Inflight_service" and h "Baggage_handling" corr= 0.645941
# 
# Since no pair has corr. = 1 it's mean no multicollinearity.
# we will keep the features

# In[204]:


data.drop(['id','Departure_Arrival_time_convenient', 'Gate_location', 'Departure_Delay_in_Minutes','Arrival_Delay_in_Minutes'] , inplace =True ,axis=1)


# # Part 2: Modeling

# # Linear Regression ,Ridge Regression and PolynomialFeatures

# In[205]:


X = data.drop(['satisfied'],axis=1)
y= data['satisfied']

X, X_test, y, y_test = train_test_split(X, y, test_size=.2, random_state=50) #hold out 20% of the data for final testing

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.25, random_state=10)

#set up the 3 models we're choosing from:

lm = LinearRegression()

#Feature scaling for train, val, and test so that we can run our ridge model on each
scaler = StandardScaler()

#Since this scaling is part of our model, 
#we need to scale using the training set feature distributions and apply the same scaling to validation and test 
#without refitting the scaler. 
X_train_scaled = scaler.fit_transform(X_train.values) 
# fit_transform: the model will learn the mean and std of all training set features


X_val_scaled = scaler.transform(X_val.values) 
#transform: transforming all the features by using mean and std, 
#we use the same the mean and std, by mean we are not calculating them
#just applying from training data , not learning
# why> to prevent data lackage , let our model learn about teast and train

X_test_scaled = scaler.transform(X_test.values)

lm_reg = Ridge(alpha=0.5)

#Feature transforms for train, val, and test so that we can run our poly model on each
poly = PolynomialFeatures(degree=2) 

X_train_poly = poly.fit_transform(X_train.values)
X_val_poly = poly.transform(X_val.values)
X_test_poly = poly.transform(X_test.values)

lm_poly = LinearRegression()

#validate

lm.fit(X_train, y_train)
print(f'Linear Regression val R^2: {lm.score(X_val, y_val):.3f}')

lm_reg.fit(X_train_scaled, y_train)
print(f'Ridge Regression val R^2: {lm_reg.score(X_val_scaled, y_val):.3f}')

lm_poly.fit(X_train_poly, y_train)
print(f'Degree 2 polynomial regression val R^2: {lm_poly.score(X_val_poly, y_val):.3f}')


# # Decision Tree

# In[206]:


X = data.drop('satisfied',axis=1)
y = data['satisfied']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=101)


# In[207]:



dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)


# In[208]:


predictions = dtree.predict(X_test)
print(classification_report(y_test,predictions))


# In[209]:


print(confusion_matrix(y_test,predictions))


# # Random Forest 

# In[210]:


X = data.drop('satisfied',axis=1)
y = data['satisfied']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=101)


# In[211]:


rfc = RandomForestClassifier(n_estimators=600)


# In[212]:


rfc.fit(X_train,y_train)


# In[213]:


predictions = rfc.predict(X_test)


# In[214]:


print(classification_report(y_test,predictions))


# In[215]:


print(confusion_matrix(y_test,predictions))  


# In[216]:


plot_roc_curve(rfc, X_test, y_test)   


# Random Forest has performed very well on both Accuracy and area under ROC curve. 

# ## Based on the score metric, Random forest classifier is the best models. 

# #  Q: Which service should airlines focus on that increase customer satisfaction?  

# The usual way to compute the feature importance values of a single tree is as follows:
# 
# 1- initialize an array feature_importances with Xs
# 
# 2- you traverse the tree: for each internal node that splits on feature i you compute the error reduction of that node multiplied by the number of samples that were routed to the node and add this quantity to feature_importances

# In[217]:


importances = rfc.feature_importances_
std = np.std([rfc.feature_importances_ for tree in rfc.estimators_], axis=0)


# feature_importances_: 
# The impurity-based feature importances.
# The higher, the more important the feature.
# (there is a high clear cut distinction)
# It is also known as the Gini importance.

# Maximize info gain = reduce Gini impurity

# In[218]:


feature_names = X.columns.tolist()
forest_importances = pd.Series(importances, index=feature_names)
indices = np.argsort(importances)
fig, ax = plt.subplots(figsize=(12, 10))
forest_importances.plot.barh(yerr=std, ax=ax ,cmap='spring' )
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()


# # A: the most important services are:
# ### 1- Online_boarding
# ### 2- Inflight_wifi_service
# ### 3- Inflight entertainment
# 
# ## another passenger criteria affect on their satisfactions
# ### 3- Travel Type
# ### 4- Class

# # Q: Do Class and Travel Type play a role in passenger satisfactions?
# # A: Yes
