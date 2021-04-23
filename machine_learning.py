#we must perform these steps while working on problems related to linear regression
#1st step is importing the data
#check no. of rows and columns
#check head and tail
#check the datatype of columns
#type conversion(change the datatype if required)
#find the descriptive statistics of numeric column
#perform EDA to understand data, mix, pattern etc/.
#Check for missing values in the data
#input missing values if any
#check for outliers in the data
#outlier treatment
#feature engineering (replace, derive new column, manipulate)
#feature transformation
#create master data by removing unnecessary columns
#seperate target variable and store it to y
#store rest of the variables into x
#split x and y into training and test samples
#train the model using training sample
#test the model output using test sample
#check model accuracy

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model
by=pd.read_table("/Users/bhaskaryuvaraj/Desktop/Downloads/House_PricesTAB_Delimited.txt")

#number of rows and columns
len(by)
len(by.columns)
#rows=128, Columns=8
by.columns

#head and tail
by.head()
by.tail()

#checking the datatypes of columns
by.dtypes
#the datatypes are correct

#to find the descriptive statistics
by.describe()
by['Price'].describe()
#since Price column is the dependent column it is seen that mean is greater than median hence it is left scewed
#count       128.000000
#mean     130427.343750
#std       26868.770371
#min       69100.000000
#25%      111325.000000
#50%      125950.000000
#75%      148250.000000
#max      211200.000000

by['SqFt'].describe() #over here mean=median and its normal distribution
by['Bedrooms'].describe() #mean is almost equal to median hence noraml distribution
by['Bathrooms'].describe() #again same as above
by['Offers'].describe()

by['Neighborhood'].unique()

#----------------------------EDA starts--------------------------------------

#count of houses by neighborhood 
#per sqft price by neighborhood
#type of houses by neighborhood contributing max revenue
#type of houses by bedrooms contributing max revenue
#count of houses by bricktype
#average price by offers
#prices by bricktype

#count of houses by neighborhood 
by.groupby('Neighborhood')['Neighborhood'].count()
#Neighborhood
#East     45
#North    45
#West     39

#per sqft price by neighborhood
by.groupby('Neighborhood')['Price'].sum()/by.groupby('Neighborhood')['SqFt'].sum()
Neighborhood
#East     62.180294
#North    57.579338
#West     76.555761
# the above numbers are the price/sqft area according to neighborhood

#type of houses by neighborhood contributing max revenue
by.groupby(['Neighborhood','Brick'])['Price'].sum()/by.groupby(['Neighborhood','Brick'])['SqFt'].sum()
#Neighborhood  Brick
#East          No       58.829746
#              Yes      66.698627
#North         No       56.479411
#              Yes      63.784615
#West          No       71.488782
#              Yes      83.777645

#type of houses by bedrooms contributing max revenue
by.groupby('Bedrooms')['Price'].sum()/by.groupby('Bedrooms')['SqFt'].sum()
#Bedrooms
#2    61.956639
#3    63.029566
#4    72.028659
#5    74.201313

#type of houses by bathrooms contributing max revenue
by.groupby('Bathrooms')['Price'].sum()/by.groupby('Bathrooms')['SqFt'].sum()
#Bathrooms
#2    62.285423
#3    68.058972
#4    87.117904

#count of houses by bricktype
by.groupby('Brick')["Brick"].count()
#Brick
#No     87
#Yes    42

#avg price by offers
by.groupby('Offers')['Price'].mean()
#Offers
#1    145239.130435
#2    132861.111111
#3    126693.478261
#4    121250.000000
#5    117533.333333
#6     90300.000000




#-----------------------------EDA ends--------------------------------------

#to find missing values
by.isnull().sum()
#there are no missing values
#if there was missing valuesin the data then we could have used one of these method
#1) delete rows having missing value
by=by.dropna()
#2) replace missing values with mean or median
by['SqFt'].mean()
by['SqFt'].median()
by['SqFt']=by['SqFt'].fillna(by['SqFt'].mean())
by['SqFt']=by['SqFt'].fillna(by['SqFt'].median())

by['Neighborhood']=by['Neighborhood'].fillna('North')

#to check outliers
plt.boxplot(by['Price']) #has outlier
plt.boxplot(by['SqFt']) #has outlier

#outlier treatement
#we need to write user defined function to remove the rows having outlier
def remove_outlier(d,c):
    #find Q1
    q1=d[c].quantile(0.25)
    q3=d[c].quantile(0.75)
    #find interquartile range
    iqr=q3-q1
    ub=q3+1.53*iqr
    lb=q1-1.53*iqr
    #filter data btw lb and ub
    result=d[(d[c]>lb) & (d[c]<ub)]
    return result

#remove outlier from price
by=remove_outlier(by,'Price')
plt.boxplot(by['Price'])
by=remove_outlier(by,'SqFt')
plt.boxplot(by['SqFt'])

#function to replace outlier value with ub and lb
#def replace_outlier(d,c):
#    #find Q1
#    q1=d[c].quantile(0.25)
#    q3=d[c].quantile(0.75)
#    #find interquartile range
#    iqr=q3-q1
#    ub=q3+1.53*iqr
#    lb=q1-1.53*iqr
#    #filter data btw lb and ub
#    if d[c]<lb:
#        d[c]=lb
#    elif d[c]>ub:
#        d[c]=ub
#    return
##
#by=replace_outlier(by,'Price')
    
#feature engineering
#use feature transformation method to create dummy variables of brick and neighborhood
#this is done to transfer character data to numeric data
dummy1=pd.get_dummies(by['Brick'])
dummy2=pd.get_dummies(by['Neighborhood'])
#combine dummie 1 and dummie 2 to by
combined_data=pd.concat([by,dummy1,dummy2],axis=1)
#remove unnecessary columns from combined data frame to create the final data

masterdata=combined_data.drop(combined_data.columns[[0,6,7]],axis=1)

#seperate target variable and store it to y
#store rest of the variables into x
y=masterdata['Price'].copy()
x=masterdata.drop(masterdata.columns[0],axis=1)

#create training and test data by splitting x and y into 70:30 ratio
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

#fit the model
lm=linear_model.LinearRegression()
#creating the model object

#create a training model
model=lm.fit(x_train,y_train)

#check the accuracy of training model
print(model.score(x_train,y_train))

#test the model using test data
#predict the price using x test
pred_y=lm.predict(x_test)

#check the prediction accuracy
print(model.score(x_test,y_test))
bhaskaryuvraj0712@gmail.com



