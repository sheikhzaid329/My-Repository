import pandas as pd
import matplotlib.pyplot as plt
import numpy as np



"""Multiple Linear Regression"""
"""This is a real estate regression problem where multiple linear regression
is used to predict property prices based on multiple features."""



# Loading the dataset into a DataFrame

DF= pd.read_csv("FinalAssessment/Section-A-Q1-USA-Real-Estate-Dataset-realtor-data-Rev1.csv")

print("DF.head():  \n",DF.head())

print("DF.shape():  \n",DF.shape)

print("DF.describe():    \n",DF.describe().round(2).T)

# There are missing values in the dataset, so we remove them using dropna()
DF = DF.dropna()

import seaborn as sns # Convention alias for Seaborn

# 'street' is encoded numerically in this dataset
Features = ['bed','bath','street','house_size']

# Analyzing the impact of these features on property prices

for i in Features:
    plt.figure()
    sns.regplot(x=i, y='price', data=DF).set(title=f'Regression plot of {i} and Price');
    plt.show()

"""We can also calculate the correlation of the new variables, this time using Seaborn's heatmap() to help us spot the strongest and weaker correlations based on warmer (reds) and cooler (blues) tones:"""
# Column Status is causing error so we will drop it 
DF=DF.drop(['status','city','state', 'prev_sold_date'],axis=1)
correlations = DF.corr()
print("correlations...\n" , correlations)
#annot = true displays the correlation values
s = sns.heatmap(correlations, annot=True).set(title='Heat Map of Real Estate Data - Pearson Correlation')
# Display the Plot
plt.show()

"""Preparing the Data
Following what has been done with the simple linear regression, after loading and exploring the data, we can divide it into features and targets. The main difference is that now our features have 4 columns instead of one.

We can use double brackets [[ ]] to select them from the dataframe:"""

y = DF['price']
X = DF[['bed', 'bath',
       'house_size', 'street']]

SEED = 200

# After defining the input features (X) and target variable (y), the dataset will be split into training and testing sets to train the model.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                     test_size=0.05,
                                                     random_state=SEED)

# After splitting the data, we can train the multiple regression model. Note that there is no need to reshape the input features (X), as they already consist of multiple dimensions.
print('X.shape # (48,4):    \n', X.shape)

#To train the model, the fit() method of the LinearRegression class is used:

from sklearn.linear_model import LinearRegression
Regressor = LinearRegression()

Regressor.fit(X_train,y_train)

#After fitting the model and finding our optimal solution, we can also look at the intercept:
print("Regressor.intercept_......\n", Regressor.intercept_)

#And at the coefficients of the features
print("Regressor.coef_ " , Regressor.coef_)

Feature_Names = X.columns
Model_Coefficients = Regressor.coef_

Coefficients_DF = pd.DataFrame(data = Model_Coefficients, 
                              index = Feature_Names, 
                              columns = ['Coefficient value'])
print(Coefficients_DF)

# let's predict with the test data:
y_pred = Regressor.predict(X_test)

Results = pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
print("Actual vs Predicted.......\n", Results)

from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error,explained_variance_score

"""sklearn.metrics
Score functions, performance metrics, pairwise metrics and distance computations.

Luckily, we don't have to do any of the metrics calculations manually. The Scikit-Learn package already comes with functions that can be used to find out the values 
of these metrics for us. Let's find the values for these metrics using our test data. First, we will import the necessary modules for calculating the MAE and MSE errors. Respectively, the Median_absolute_error and mean_squared_error:
"""

"""After exploring, training and looking at our model predictions - our final step is to evaluate the performance of our multiple linear regression. We want to understand if our predicted values are too far from our actual values. We'll do this in the same way we had previously done, by calculating the MAE, MSE and RMSE metrics.
Therefore, the regression model will be evaluated using RMSE, MAPE, and the Explained Variance Score (EVS)"""
# Calculating the evaluation metrics
RMSE = root_mean_squared_error(y_test, y_pred)
MAPE = mean_absolute_percentage_error(y_test, y_pred)
EVS = explained_variance_score(y_test, y_pred)

print(f'Root_mean_squared_error: {RMSE:.2f}')
print(f'Mean_absolute_percentage_error: {MAPE:.2f}')
print(f'Explained_variance_score: {EVS:.2f}')


""""To dig further into what is happening to our model, we can look at a metric that measures the model in a different way, it doesn't consider our individual data values such as MSE, RMSE and MAE, but takes a more general approach to the error, the R2:"""
# Manual R2 calculation (for verification)

Actual_minus_predicted = sum((y_test - y_pred)**2)
Actual_minus_actual_mean = sum((y_test - y_pred.mean())**2)
R2 = 1 - Actual_minus_predicted/Actual_minus_actual_mean
print('R2: ', R2)
# Using sklearn to calculate R2
"""The R2 doesn't tell us about how far or close each predicted value is from the real data - it tells us how much of our target is being captured by our model."""

"""R2 also comes implemented by default into the score method of Scikit-Learn's linear regressor class. We can calculate it like this:
"""
print(" R2 also comes implemented by default into the score method of Scikit-Learn's linear regressor class...\n", Regressor.score(X_test, y_test))
# End of Multiple Linear Regression