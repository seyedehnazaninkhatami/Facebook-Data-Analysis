## Nazanin
## Importing Classes 

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from sklearn import cross_validation
from sklearn.cross_validation import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn import preprocessing
import time
import csv


## Loading Data
#def read_data_fb():
print('Reading facebook dataset ...')
train_x = np.loadtxt('Desktop\MachineLearning\HW04\Data\data.csv', delimiter=',')
train_y = np.loadtxt('Desktop\MachineLearning\HW04\Data\labels.csv', delimiter=',')
kaggle_x = np.loadtxt('Desktop\MachineLearning\HW04\Data\kaggle_data.csv', delimiter=',')



## Split the provided data into training and test sets (80% train data, 20% test data)
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

####################################### Question 1 ###########################################

## DTR with max_depth 3
StartTime = time.time()
# Model Fitting
DTR_one = DecisionTreeRegressor(max_depth=3, random_state=0, criterion="mae")
DTR_one_fit = DTR_one.fit(train_x, train_y)
# Cross Validation
DTR_one_scores = cross_val_score(DTR_one_fit, train_x, train_y, cv = 5, scoring=('neg_mean_absolute_error'))
# Run Time Calculation
DTR_one_CVTime = time.time() - StartTime
# Predicting Test Data
DTR_one_prediction = DTR_one_fit.predict(test_x)
# Calculating Test Error
Test_error_DTR_one = mean_absolute_error(test_y, DTR_one_prediction)

print("mean cross validation score: {}".format(np.mean(DTR_one_scores)))
print("score without cv: {}".format(DTR_one_fit.score(train_x, train_y)))
print("DTR with with max_depth 3 MAE:{}" .format(DTR_one_scores))
print("cross validation runtime DTR with max_depth 3 :{}" .format(DTR_one_CVTime))
print("test error with max_depth 3: {}" .format(Test_error_DTR_one))



## DTR with max_depth 6
StartTime = time.time()
# Model Fitting
DTR_two = DecisionTreeRegressor(max_depth=6, random_state=0, criterion="mae")
DTR_two_fit = DTR_two.fit(train_x, train_y)
# Cross Validation
DTR_two_scores = cross_val_score(DTR_two_fit, train_x, train_y, cv = 5, scoring=('neg_mean_absolute_error'))
# Run Time Calculation
DTR_two_CVTime = time.time() - StartTime
# Predicting Test Data
DTR_two_prediction = DTR_two_fit.predict(test_x)
# Calculating Test Error
Test_error_DTR_two = mean_absolute_error(test_y, DTR_two_prediction)

print("mean cross validation score: {}".format(np.mean(DTR_two_scores)))
print("score without cv: {}".format(DTR_two_fit.score(train_x, train_y)))
print("DTR with with max_depth 6 MAE:{}" .format(DTR_two_scores))
print("cross validation runtime DTR with max_depth 6 :{}" .format(DTR_two_CVTime))
print("test error with max_depth 6: {}" .format(Test_error_DTR_two))



## DTR with max_depth 9
StartTime = time.time()
# Model Fitting
DTR_three = DecisionTreeRegressor(max_depth=9, random_state=0, criterion="mae")
DTR_three_fit = DTR_three.fit(train_x, train_y)
# Cross Validation
DTR_three_scores = cross_val_score(DTR_three_fit, train_x, train_y, cv = 5, scoring=('neg_mean_absolute_error'))
# Run Time Calculation
DTR_three_CVTime = time.time() - StartTime
# Predicting Test Data
DTR_three_prediction = DTR_three_fit.predict(test_x)
# Calculating Test Error
Test_error_DTR_three = mean_absolute_error(test_y, DTR_three_prediction)

print("mean cross validation score: {}".format(np.mean(DTR_three_scores)))
print("score without cv: {}".format(DTR_three_fit.score(train_x, train_y)))
print("DTR with with max_depth 9 MAE:{}" .format(DTR_three_scores))
print("cross validation runtime DTR with max_depth 9 :{}" .format(DTR_three_CVTime))
print("cross validation runtime DTR with max_depth 9 :{}" .format(DTR_three_CVTime))
print("test error with max_depth 9: {}" .format(Test_error_DTR_three))



## DTR with max_depth 12
StartTime = time.time()
# Model Fitting
DTR_four = DecisionTreeRegressor(max_depth=12, random_state=0, criterion="mae")
DTR_four_fit = DTR_four.fit(train_x, train_y)
# Cross Validation
DTR_four_scores = cross_val_score(DTR_four_fit, train_x, train_y, cv = 5, scoring=('neg_mean_absolute_error'))
# Run Time Calculation
DTR_four_CVTime = time.time() - StartTime
# Predicting Test Data
DTR_four_prediction = DTR_four_fit.predict(test_x)
# Calculating Test Error
Test_error_DTR_four = mean_absolute_error(test_y, DTR_four_prediction)

print("mean cross validation score: {}".format(np.mean(DTR_four_scores)))
print("score without cv: {}".format(DTR_four_fit.score(train_x, train_y)))
print("DTR with with max_depth 12 MAE:{}" .format(DTR_four_scores))
print("cross validation runtime DTR with max_depth 12 :{}" .format(DTR_four_CVTime))
print("test error with max_depth 12: {}" .format(Test_error_DTR_four))



## DTR with max_depth 15
StartTime = time.time()
# Model Fitting
DTR_five = DecisionTreeRegressor(max_depth=15, random_state=0, criterion="mae")
DTR_five_fit = DTR_five.fit(train_x, train_y)
# Cross Validation
DTR_five_scores = cross_val_score(DTR_five_fit, train_x, train_y, cv = 5, scoring=('neg_mean_absolute_error'))
# Run Time Calculation
DTR_five_CVTime = time.time() - StartTime
# Predicting Test Data
DTR_five_prediction = DTR_five_fit.predict(test_x)
# Calculating Test Error
Test_error_DTR_five = mean_absolute_error(test_y, DTR_five_prediction)

print("mean cross validation score: {}".format(np.mean(DTR_five_scores)))
print("score without cv: {}".format(DTR_five_fit.score(train_x, train_y)))
print("DTR with with max_depth 15 MAE:{}" .format(DTR_five_scores))
print("cross validation runtime DTR with max_depth 15 :{}" .format(DTR_five_CVTime))
print("test error with max_depth 15: {}" .format(Test_error_DTR_five))

## Plotting the time that it takes to perform cross-validation with each model 

clf()
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
x = [1, 2, 3, 4, 5]
y = [DTR_one_CVTime, DTR_two_CVTime, DTR_three_CVTime, DTR_four_CVTime, DTR_five_CVTime]
plt.plot([x,y, 'ro')

plt.xlabel('Model')
plt.ylabel('CrossValidation Time')
plt.title('CrossValidation Time')
plt.savefig('Desktop\MachineLearning\HW04\Submission\Figures')


################################## Question 2 ########################################


## KNR with 3 Neighbors
StartTime = time.time()
# Model Fitting
KNR_one = KNeighborsRegressor(n_neighbors=3)
KNR_one_fit = KNR_one.fit(train_x, train_y)
# Cross Validation
KNR_one_scores = cross_val_score(KNR_one_fit, train_x, train_y, cv = 5, scoring=('neg_mean_absolute_error'))
# Runtime Calculation
KNR_one_CVTime = time.time() - StartTime
# Predicting Test Data
KNR_one_prediction = KNR_one_fit.predict(test_x)
# Calculating Test Error
Test_error_KNR_one = mean_absolute_error(test_y, KNR_one_prediction)

print("mean cross validation score: {}".format(np.mean(KNR_one_scores)))
print("score without cv: {}".format(KNR_one_fit.score(train_x, train_y)))
print("KNR with 3 neighbors MAE:{}" .format(KNR_one_scores))
print("cross validation runtime KNR with 3 neighbors :{}" .format(KNR_one_CVTime))
print("test error with 3 Neighbors: {}" .format(Test_error_KNR_one))


## KNR with 5 Neighbors
StartTime = time.time()
# Model Fitting
KNR_two = KNeighborsRegressor(n_neighbors=5)
KNR_two_fit = KNR_two.fit(train_x, train_y)
# Cross Validation
KNR_two_scores = cross_val_score(KNR_two_fit, train_x, train_y, cv = 5, scoring=('neg_mean_absolute_error'))
# Runtime Calculation
KNR_two_CVTime = time.time() - StartTime
# Predicting Test Data
KNR_two_prediction = KNR_two_fit.predict(test_x)
# Calculating Test Error
Test_error_KNR_two = mean_absolute_error(test_y, KNR_two_prediction)

print("mean cross validation score: {}".format(np.mean(KNR_two_scores)))
print("score without cv: {}".format(KNR_two_fit.score(train_x, train_y)))
print("KNR with 5 neighbors MAE:{}" .format(KNR_two_scores))
print("cross validation runtime KNR with 5 neighbors :{}" .format(KNR_two_CVTime))
print("test error with 5 Neighbors: {}" .format(Test_error_KNR_two))



## KNR with 10 Neighbors
StartTime = time.time()
# Model Fitting
KNR_three = KNeighborsRegressor(n_neighbors=10)
KNR_three_fit = KNR_three.fit(train_x, train_y)
# Cross Validation
KNR_three_scores = cross_val_score(KNR_three_fit, train_x, train_y, cv = 5, scoring=('neg_mean_absolute_error'))
# Runtime Calculation
KNR_three_CVTime = time.time() - StartTime
# Predicting Test Data
KNR_three_prediction = KNR_three_fit.predict(test_x)
# Calculating Test Error
Test_error_KNR_three = mean_absolute_error(test_y, KNR_three_prediction)

print("mean cross validation score: {}".format(np.mean(KNR_three_scores)))
print("score without cv: {}".format(KNR_three_fit.score(train_x, train_y)))
print("KNR with 10 neighbors MAE:{}" .format(KNR_three_scores))
print("cross validation runtime KNR with 10 neighbors :{}" .format(KNR_three_CVTime))
print("test error with 10 Neighbors: {}" .format(Test_error_KNR_three))



## KNR with 20 Neighbors
StartTime = time.time()
# Model Fitting
KNR_four = KNeighborsRegressor(n_neighbors=20)
KNR_four_fit = KNR_four.fit(train_x, train_y)
# Cross Validation
KNR_four_scores = cross_val_score(KNR_four_fit, train_x, train_y, cv = 5, scoring=('neg_mean_absolute_error'))
# Runtime Calculation
KNR_four_CVTime = time.time() - StartTime
# Predicting Test Data
KNR_four_prediction = KNR_four_fit.predict(test_x)
# Calculating Test Error
Test_error_KNR_four = mean_absolute_error(test_y, KNR_four_prediction)

print("mean cross validation score: {}".format(np.mean(KNR_four_scores)))
print("score without cv: {}".format(KNR_four_fit.score(train_x, train_y)))
print("KNR with 20 neighbors MAE:{}" .format(KNR_four_scores))
print("cross validation runtime KNR with 20 neighbors :{}" .format(KNR_four_CVTime))
print("test error with 20 Neighbors: {}" .format(Test_error_KNR_four))



## KNR with 25 Neighbors
StartTime = time.time()
# Model Fitting
KNR_five = KNeighborsRegressor(n_neighbors=25)
KNR_five_fit = KNR_five.fit(train_x, train_y)
# Cross Validation
KNR_five_scores = cross_val_score(KNR_five_fit, train_x, train_y, cv = 5, scoring=('neg_mean_absolute_error'))
# Runtime Calculation
KNR_five_CVTime = time.time() - StartTime
# Predicting Test Data
KNR_five_prediction = KNR_five_fit.predict(test_x)
# Calculating Test Error
Test_error_KNR_five = mean_absolute_error(test_y, KNR_five_prediction)

print("mean cross validation score: {}".format(np.mean(KNR_five_scores)))
print("score without cv: {}".format(KNR_five_fit.score(train_x, train_y)))
print("KNR with 25 neighbors MAE:{}" .format(KNR_five_scores))
print("cross validation runtime KNR with 25 neighbors :{}" .format(KNR_five_CVTime))
print("test error with 25 Neighbors: {}" .format(Test_error_KNR_five))


## Question 2 Full trainset

## KNR with 5 Neighbors
StartTime = time.time()
# Model Fitting
KNR_two = KNeighborsRegressor(n_neighbors=5)
KNR_two_fit = KNR_two.fit(train_x, train_y)
# Cross Validation
KNR_two_scores_fulltrain = cross_val_score(KNR_two_fit, train_x, train_y, cv = None , scoring=('neg_mean_absolute_error'))
# Runtime Calculation
KNR_two_CVTime = time.time() - StartTime
# Predicting Test Data
KNR_two_prediction = KNR_two_fit.predict(test_x)
# Calculating Test Error
Test_error_KNR_two_fulltrain = mean_absolute_error(test_y, KNR_two_prediction)

print("mean cross validation score: {}".format(KNR_two_scores_fulltrain))
print("score without cv: {}".format(KNR_two_fit.score(train_x, train_y)))
print("KNR with 5 neighbors MAE:{}" .format(KNR_two_scores_fulltrain))
print("cross validation runtime KNR with 5 neighbors :{}" .format(KNR_two_CVTime))
print("test error with 5 Neighbors: {}" .format(Test_error_KNR_two_fulltrain))


## Question 2c
## KNR with 5 Neighbors and Manhattan Distance Measurement
StartTime = time.time()
# Model Fitting
KNR_two = KNeighborsRegressor(n_neighbors=5, p = 1)
KNR_two_fit = KNR_two.fit(train_x, train_y)
# Cross Validation
KNR_two_scores = cross_val_score(KNR_two_fit, train_x, train_y, cv = 5, scoring=('neg_mean_absolute_error'))
# Runtime Calculation
KNR_two_CVTime = time.time() - StartTime
# Predicting Test Data
KNR_two_prediction = KNR_two_fit.predict(test_x)
# Calculating Test Error
Test_error_KNR_two = mean_absolute_error(test_y, KNR_two_prediction)

print("mean cross validation score with Manhattan distance: {}".format(np.mean(KNR_two_scores)))
print("score without cv with Manhattan distance: {}".format(KNR_two_fit.score(train_x, train_y)))
print("KNR with 5 neighbors MAE with Manhattan distance:{}" .format(KNR_two_scores))
print("cross validation runtime KNR with 5 neighbors with Manhattan distance:{}" .format(KNR_two_CVTime))
print("test error with 5 Neighbors with Manhattan distance: {}" .format(Test_error_KNR_two))

## KNR with 5 Neighbors and Arbitrarily Minkowski Distance Measurement
StartTime = time.time()
# Model Fitting
KNR_two = KNeighborsRegressor(n_neighbors=5, p = 100)
KNR_two_fit = KNR_two.fit(train_x, train_y)
# Cross Validation
KNR_two_scores = cross_val_score(KNR_two_fit, train_x, train_y, cv = 5, scoring=('neg_mean_absolute_error'))
# Runtime Calculation
KNR_two_CVTime = time.time() - StartTime
# Predicting Test Data
KNR_two_prediction = KNR_two_fit.predict(test_x)
# Calculating Test Error
Test_error_KNR_two = mean_absolute_error(test_y, KNR_two_prediction)

print("mean cross validation score with Minkowski distance: {}".format(np.mean(KNR_two_scores)))
print("score without cv with Minkowski distance: {}".format(KNR_two_fit.score(train_x, train_y)))
print("KNR with 5 neighbors MAE with Minkowski distance:{}" .format(KNR_two_scores))
print("cross validation runtime KNR with 5 neighbors with Minkowski distance:{}" .format(KNR_two_CVTime))
print("test error with 5 Neighbors with Minkowski distance: {}" .format(Test_error_KNR_two))

################################# Question 3 ################################

## Linear Ridge with Regularization Constant 10 ** -6
StartTime = time.time()
# Model Fitting
Ridge_one = linear_model.Ridge(alpha = 0.000001,normalize = True)
Ridge_one_fit = Ridge_one.fit(train_x, train_y)
# Cross Validation
Ridge_one_scores = cross_val_score(Ridge_one_fit, train_x, train_y, cv = 5, scoring=('neg_mean_absolute_error'))
# Runtime Calculation
Ridge_one_CVTime = time.time() - StartTime
# Predicting Test Data
Ridge_one_prediction = Ridge_one_fit.predict(test_x)
# Calculating Test Error
Test_error_Ridge_one = mean_absolute_error(test_y, Ridge_one_prediction)

print("Ridge with alpha 10^-6 cross validation scores:{}".format(np.mean(Ridge_one_scores)))
print("score without cv: {}".format(Ridge_one_fit.score(train_x, train_y)))
print("cross validation runtime Ridge with alpha 10^-6 :{}" .format(Ridge_one_CVTime))
print("test error with Reg constat 10 ** -6: {}" .format(Test_error_Ridge_one))




## Linear Ridge with Regularization Constant 10 ** -4
StartTime = time.time()
# Model Fitting
Ridge_two = linear_model.Ridge(alpha = 0.0001,normalize = True)
Ridge_two_fit = Ridge_two.fit(train_x, train_y)
# Cross Validation
Ridge_two_scores = cross_val_score(Ridge_two_fit, train_x, train_y, cv = 5, scoring=('neg_mean_absolute_error'))
# Runtime Calculation
Ridge_two_CVTime = time.time() - StartTime
# Predicting Test Data
Ridge_two_prediction = Ridge_two_fit.predict(test_x)
# Calculating Test Error
Test_error_Ridge_two = mean_absolute_error(test_y, Ridge_two_prediction)

print("Ridge with alpha 10^-4 cross validation scores:{}".format(np.mean(Ridge_two_scores)))
print("score without cv: {}".format(Ridge_two_fit.score(train_x, train_y)))
print("cross validation runtime Ridge with alpha 10^-4 :{}" .format(Ridge_two_CVTime))
print("test error with Reg constat 10 ** -4: {}" .format(Test_error_Ridge_two))


## Linear Ridge with Regularization Constant 10 ** -2
StartTime = time.time()
# Model Fitting
Ridge_three = linear_model.Ridge(alpha = 0.01,normalize = True)
Ridge_three_fit = Ridge_three.fit(train_x, train_y)
# Cross Validation
Ridge_three_scores = cross_val_score(Ridge_three_fit, train_x, train_y, cv = 5, scoring=('neg_mean_absolute_error'))
# Runtime Calculation
Ridge_three_CVTime = time.time() - StartTime
# Predicting Test Data
Ridge_three_prediction = Ridge_three_fit.predict(test_x)
# Calculating Test Error
Test_error_Ridge_three = mean_absolute_error(test_y, Ridge_three_prediction)

print("Ridge with alpha 10^-2 cross validation scores:{}".format(np.mean(Ridge_three_scores)))
print("score without cv: {}".format(Ridge_three_fit.score(train_x, train_y)))
print("cross validation runtime Ridge with alpha 10^-2 :{}" .format(Ridge_three_CVTime))
print("test error with Reg constat 10 ** -2: {}" .format(Test_error_Ridge_three))



## Linear Ridge with Regularization Constant 1
StartTime = time.time()
# Model Fitting
Ridge_four = linear_model.Ridge(alpha = 1,normalize = True)
Ridge_four_fit = Ridge_four.fit(train_x, train_y)
# Cross Validation
Ridge_four_scores = cross_val_score(Ridge_four_fit, train_x, train_y, cv = 5, scoring=('neg_mean_absolute_error'))
# Runtime Calculation
Ridge_four_CVTime = time.time() - StartTime
# Predicting Test Data
Ridge_four_prediction = Ridge_four_fit.predict(test_x)
# Calculating Test Error
Test_error_Ridge_four = mean_absolute_error(test_y, Ridge_four_prediction)

print("Ridge with alpha 1 cross validation scores:{}".format(np.mean(Ridge_four_scores)))
print("score without cv: {}".format(Ridge_four_fit.score(train_x, train_y)))
print("cross validation runtime Ridge with alpha 1:{}" .format(Ridge_four_CVTime))
print("test error with Reg constat 1: {}" .format(Test_error_Ridge_four))


## Linear Ridge with Regularization Constant 10
StartTime = time.time()
# Model Fitting
Ridge_five = linear_model.Ridge(alpha = 10,normalize = True)
Ridge_five_fit = Ridge_five.fit(train_x, train_y)
# Cross Validation
Ridge_five_scores = cross_val_score(Ridge_five_fit, train_x, train_y, cv = 5, scoring=('neg_mean_absolute_error'))
# Runtime Calculation
Ridge_five_CVTime = time.time() - StartTime
# Predicting Test Data
Ridge_five_prediction = Ridge_five_fit.predict(test_x)
# Calculating Test Error
Test_error_Ridge_five = mean_absolute_error(test_y, Ridge_five_prediction)

print("Ridge with alpha 10 cross validation scores:{}".format(np.mean(Ridge_five_scores)))
print("score without cv: {}".format(Ridge_five_fit.score(train_x, train_y)))
print("cross validation runtime Ridge with alpha 10:{}" .format(Ridge_five_CVTime))
print("test error with Reg constat 10: {}" .format(Test_error_Ridge_five))



## Linear Lasso with Regularization Constant 10 ** -6
StartTime = time.time()
# Model Fitting
Lasso_one = linear_model.Lasso(alpha = 0.000001,normalize = True)
Lasso_one_fit = Lasso_one.fit(train_x, train_y)
# Cross Validation
Lasso_one_scores = cross_val_score(Lasso_one_fit, train_x, train_y, cv = 5, scoring=('neg_mean_absolute_error'))
# Runtime Calculation
Lasso_one_CVTime = time.time() - StartTime
# Predicting Test Data
Lasso_one_prediction = Lasso_one_fit.predict(test_x)
# Calculating Test Error
Test_error_Lasso_one = mean_absolute_error(test_y, Lasso_one_prediction)

print("Lasso with alpha 10^-6 cross validation scores:{}".format(np.mean(Lasso_one_scores)))
print("score without cv: {}".format(Lasso_one_fit.score(train_x, train_y)))
print("Lasso with alpha 10^ -6 MAE:{}" .format(Lasso_one_scores))
print("cross validation runtime Lasso with alpha 10^-6 :{}" .format(Lasso_one_CVTime))
print("test error with Reg constat 10 ** -6: {}" .format(Test_error_Lasso_one))




## Linear Lasso with Regularization Constant 10 ** -4
StartTime = time.time()
# Model Fitting
Lasso_two = linear_model.Lasso(alpha = 0.0001,normalize = True)
Lasso_two_fit = Lasso_two.fit(train_x, train_y)
# Cross Validation
Lasso_two_scores = cross_val_score(Lasso_two_fit, train_x, train_y, cv = 5, scoring=('neg_mean_absolute_error'))
# Runtime Calculation
Lasso_two_CVTime = time.time() - StartTime
# Predicting Test Data
Lasso_two_prediction = Lasso_two_fit.predict(test_x)
# Calculating Test Error
Test_error_Lasso_two = mean_absolute_error(test_y, Lasso_two_prediction)

print("Lasso with alpha 10^-4 cross validation scores:{}".format(np.mean(Lasso_two_scores)))
print("score without cv: {}".format(Lasso_two_fit.score(train_x, train_y)))
print("Lasso with alpha 10^ -4 MAE:{}" .format(Lasso_two_scores))
print("cross validation runtime Lasso with alpha 10^-4 :{}" .format(Lasso_two_CVTime))
print("test error with Reg constat 10 ** -4: {}" .format(Test_error_Lasso_two))


## Linear Lasso with Regularization Constant 10 ** -2
StartTime = time.time()
# Model Fitting
Lasso_three = linear_model.Lasso(alpha = 0.01,normalize = True)
Lasso_three_fit = Lasso_three.fit(train_x, train_y)
# Cross Validation
Lasso_three_scores = cross_val_score(Lasso_three_fit, train_x, train_y, cv = 5, scoring=('neg_mean_absolute_error'))
# Runtime Calculation
Lasso_three_CVTime = time.time() - StartTime
# Predicting Test Data
Lasso_three_prediction = Lasso_three_fit.predict(test_x)
# Calculating Test Error
Test_error_Lasso_three = mean_absolute_error(test_y, Lasso_three_prediction)

print("Lasso with alpha 10^-2 cross validation scores:{}".format(np.mean(Lasso_three_scores)))
print("score without cv: {}".format(Lasso_three_fit.score(train_x, train_y)))
print("Lasso with alpha 10^ -2 MAE:{}" .format(Lasso_three_scores))
print("cross validation runtime Lasso with alpha 10^-2 :{}" .format(Lasso_three_CVTime))
print("test error with Reg constat 10 ** -2: {}" .format(Test_error_Lasso_three))



## Linear Lasso with Regularization Constant 1
StartTime = time.time()
# Model Fitting
Lasso_four = linear_model.Lasso(alpha = 1,normalize = True)
Lasso_four_fit = Lasso_four.fit(train_x, train_y)
# Cross Validation
Lasso_four_scores = cross_val_score(Lasso_four_fit, train_x, train_y, cv = 5, scoring=('neg_mean_absolute_error'))
# Runtime Calculation
Lasso_four_CVTime = time.time() - StartTime
# Predicting Test Data
Lasso_four_prediction = Lasso_four_fit.predict(test_x)
# Calculating Test Error
Test_error_Lasso_four = mean_absolute_error(test_y, Lasso_four_prediction)

print("Lasso with alpha 1 cross validation scores:{}".format(np.mean(Lasso_four_scores)))
print("score without cv: {}".format(Lasso_four_fit.score(train_x, train_y)))
print("Lasso with alpha 1 MAE:{}" .format(Lasso_four_scores))
print("cross validation runtime Lasso with alpha 1:{}" .format(Lasso_four_CVTime))
print("test error with Reg constat 1: {}" .format(Test_error_Lasso_four))


## Linear Lasso with Regularization Constant 10
StartTime = time.time()
# Model Fitting
Lasso_five = linear_model.Lasso(alpha = 10,normalize = True)
Lasso_five_fit = Lasso_five.fit(train_x, train_y)
# Cross Validation
Lasso_five_scores = cross_val_score(Lasso_five_fit, train_x, train_y, cv = 5, scoring=('neg_mean_absolute_error'))
# Runtime Calculation
Lasso_five_CVTime = time.time() - StartTime
# Predicting Test Data
Lasso_five_prediction = Lasso_five_fit.predict(test_x)
# Calculating Test Error
Test_error_Lasso_five = mean_absolute_error(test_y, Lasso_five_prediction)

print("Lasso with alpha 10 cross validation scores:{}".format(np.mean(Lasso_five_scores)))
print("score without cv: {}".format(Lasso_five_fit.score(train_x, train_y)))
print("Lasso with alpha 10 MAE:{}" .format(Lasso_five_scores))
print("cross validation runtime Lasso with alpha 10:{}" .format(Lasso_five_CVTime))
print("test error with Reg constat 10: {}" .format(Test_error_Lasso_five))

## Q3 Best Performance Fulltrain
## Linear Lasso with Regularization Constant 10 ** -2
StartTime = time.time()
# Model Fitting
Lasso_three = linear_model.Lasso(alpha = 0.01,normalize = True)
Lasso_three_fit = Lasso_three.fit(train_x, train_y)
# Cross Validation
Lasso_three_scores = cross_val_score(Lasso_three_fit, train_x, train_y, cv = None, scoring=('neg_mean_absolute_error'))
# Runtime Calculation
Lasso_three_CVTime = time.time() - StartTime
# Predicting Test Data
Lasso_three_prediction = Lasso_three_fit.predict(test_x)
# Calculating Test Error
Test_error_Lasso_three = mean_absolute_error(test_y, Lasso_three_prediction)

print("Lasso with alpha 10^-2 cross validation scores:{}".format(np.mean(Lasso_three_scores)))
print("score without cv: {}".format(Lasso_three_fit.score(train_x, train_y)))
print("cross validation runtime Lasso with alpha 10^-2 :{}" .format(Lasso_three_CVTime))
print("test error with Reg constat 10 ** -2: {}" .format(Test_error_Lasso_three))



################################3 Question 4 #####################################
         
# Preprocessing data
train_x = preprocessing.scale(train_x)
#train_y = preprocessing.scale(train_y)
test_x = preprocessing.scale(test_x)
#trest_y = preprocessing.scale(test_y)

## SVM with Poly degree 1
StartTime = time.time()
# Model Fitting
SVM_one = SVR(C=1.0, epsilon=0.2, kernel='poly', degree=1)
SVM_one_fit = SVM_one.fit(train_x,train_y)
# Cross Validation
SVM_one_scores = cross_val_score(SVM_one_fit, train_x, train_y, cv = 5, scoring=('neg_mean_absolute_error'))
# Runtime Calculation
SVM_one_CVTime = time.time() - StartTime
# Predicting Test Data
SVM_one_predictions = SVM_one_fit.predict(test_x)
# Calculating Test Error
Test_error_SVM_one = mean_absolute_error(test_y, SVM_one_predictions)

print(" SVM degree 1 mean score:{}".format(np.mean(SVM_one_mean)))
print("SVM degree 1: {}".format(SVM_one_fit.score(train_x, train_y)))
print("SVM degree 1 MAE:{}" .format(SVM_one_scores))
print("cross validation runtime SVM degree 1 :{}" .format(SVM_one_predictions))
print("test error SVM degree 1: {}" .format(Test_error_SVM_one))

## SVM with Poly degree 2
StartTime = time.time()
# Model Fitting
SVM_two = SVR(C=1.0, epsilon=0.2, kernel='poly', degree=2)
SVM_two_fit = SVM_two.fit(train_x,train_y)
# Cross Validation
SVM_two_scores = cross_val_score(SVM_two_fit, train_x, train_y, cv = 5, scoring=('neg_mean_absolute_error'))
# Runtime Calculation
SVM_two_CVTime = time.time() - StartTime
# Predicting Test Data
SVM_two_predictions = SVM_two_fit.predict(test_x)
# Calculating Test Error
Test_error_SVM_two = mean_absolute_error(test_y, SVM_two_predictions)

print(" SVM degree 2 mean score:{}".format(np.mean(SVM_two_scores)))
print("SVM degree 2: {}".format(SVM_two_fit.score(train_x, train_y)))
print("SVM degree 2 MAE:{}" .format(SVM_two_scores))
print("cross validation runtime SVM degree 2 :{}" .format(SVM_two_predictions))
print("test error SVM degree 2: {}" .format(Test_error_SVM_two))

## SVM with RBF
StartTime = time.time()
# Model Fitting
SVM_three = SVR(C=1.0, epsilon=0.2, kernel='rbf', degree=1)
SVM_three_fit = SVM_three.fit(train_x,train_y)
# Cross Validation
SVM_three_scores = cross_val_score(SVM_three_fit, train_x, train_y, cv = 5, scoring=('neg_mean_absolute_error'))
# Runtime Calculation
SVM_three_CVTime = time.time() - StartTime
# Predicting Test Data
SVM_three_predictions = SVM_three_fit.predict(test_x)
# Calculating Test Error
Test_error_SVM_three = mean_absolute_error(test_y, SVM_three_predictions)

print("SVM RBF  mean score:{}".format(np.mean(SVM_three_scores)))
print("SVM RBF : {}".format(SVM_three_fit.score(train_x, train_y)))
print("SVM RBF  MAE:{}" .format(SVM_three_scores))
print("cross validation runtime SVM RBF :{}" .format(SVM_three_predictions))
print("test error SVM RBF : {}" .format(Test_error_SVM_three))



########################## Question 5 #################################
# Preprocessing data
train_x = preprocessing.scale(train_x)
train_y = preprocessing.scale(train_y)
test_x = preprocessing.scale(test_x)
trest_y = preprocessing.scale(test_y)


## NN with one hidden layer of size 10
StartTime = time.time()
# Model Fitting
MLP_one = MLPRegressor(hidden_layer_sizes=10)
MLP_one_fit = MLP_one.fit(train_x,train_y)
# Cross Validation
MLP_one_scores = cross_val_score(MLP_one_fit, train_x, train_y, cv = 5, scoring=('neg_mean_absolute_error'))
# Runtime Calculation
MLP_one_CVTime = time.time() - StartTime
# Predicting Test Data
MLP_one_predictions = MLP_one_fit.predict(test_x)
# Calculating Test Error
Test_error_MLP_one = mean_absolute_error(test_y, MLP_one_predictions)
print("NNR with a hidden layer with size 10 mean score:{}".format(np.mean(MLP_one_scores)))
print("NNR with a hidden layer with size 10 score without cv:{}".format(MLP_one_fit.score(train_x, train_y)))
print("NNR with a hidden layer with size 10 MAE:{}" .format(MLP_one_scores))
print("cross validation runtime NNR with a hidden layer with size 10:{}".format(MLP_one_CVTime))
print("test error with a hidden layer with size 10: {}" .format(Test_error_MLP_one))


## NN with one hidden layer of size 20
StartTime = time.time()
# Model Fitting
MLP_two = MLPRegressor(hidden_layer_sizes=20)
MLP_two_fit = MLP_two.fit(train_x,train_y)
# Cross Validation
MLP_two_scores = cross_val_score(MLP_two_fit, train_x, train_y, cv = 5, scoring=('neg_mean_absolute_error'))
# Runtime Calculation
MLP_two_CVTime = time.time() - StartTime
# Predicting Test Data
MLP_two_predictions = MLP_two_fit.predict(test_x)
# Calculating Test Error
Test_error_MLP_two = mean_absolute_error(test_y, MLP_two_predictions)
print("NNR with a hidden layer with size 20 mean score:{}".format(np.mean(MLP_two_scores)))
print("NNR with a hidden layer with size 20 score without cv:{}".format(MLP_two_fit.score(train_x, train_y)))
print("NNR with a hidden layer with size 20 MAE:{}" .format(MLP_two_scores))
print("cross validation runtime NNR with a hidden layer with size 20:{}".format(MLP_two_CVTime))
print("test error with a hidden layer with size 20: {}" .format(Test_error_MLP_two))


## NN with one hidden layer of size 30
StartTime = time.time()
# Model Fitting
MLP_three = MLPRegressor(hidden_layer_sizes=30)
MLP_three_fit = MLP_three.fit(train_x,train_y)
# Cross Validation
MLP_three_scores = cross_val_score(MLP_three_fit, train_x, train_y, cv = 5, scoring=('neg_mean_absolute_error'))
# Runtime Calculation
MLP_three_CVTime = time.time() - StartTime
# Predicting Test Data
MLP_three_predictions = MLP_three_fit.predict(test_x)
# Calculating Test Error
Test_error_MLP_three = mean_absolute_error(test_y, MLP_three_predictions)
print("NNR with a hidden layer with size 30 mean score:{}".format(np.mean(MLP_three_scores)))
print("NNR with a hidden layer with size 30 score without cv:{}".format(MLP_three_fit.score(train_x, train_y)))
print("NNR with a hidden layer with size 30 MAE:{}" .format(MLP_three_scores))
print("cross validation runtime NNR with a hidden layer with size 30:{}".format(MLP_three_CVTime))
print("test error with a hidden layer with size 30: {}" .format(Test_error_MLP_three))


## NN with one hidden layer of size 40
StartTime = time.time()
# Model Fitting
MLP_four = MLPRegressor(hidden_layer_sizes=40)
MLP_four_fit = MLP_four.fit(train_x,train_y)
# Cross Validation
MLP_four_scores = cross_val_score(MLP_four_fit, train_x, train_y, cv = 5, scoring=('neg_mean_absolute_error'))
# Runtime Calculation
MLP_four_CVTime = time.time() - StartTime
# Predicting Test Data
MLP_four_predictions = MLP_four_fit.predict(test_x)
# Calculating Test Error
Test_error_MLP_four = mean_absolute_error(test_y, MLP_four_predictions)
print("NNR with a hidden layer with size 40 mean score:{}".format(np.mean(MLP_four_scores)))
print("NNR with a hidden layer with size 40 score without cv:{}".format(MLP_four_fit.score(train_x, train_y)))
print("NNR with a hidden layer with size 40 MAE:{}" .format(MLP_four_scores))
print("cross validation runtime NNR with a hidden layer with size 40:{}".format(MLP_four_CVTime))
print("test with a hidden layer with size 40: {}" .format(Test_error_MLP_four))



######################################## Question 6 ###############################################
## Kaggle Competition
## Model Selection


##
def kaggleize(predictions,file):

	if(len(predictions.shape)==1):
		predictions.shape = [predictions.shape[0],1]

	ids = 1 + np.arange(predictions.shape[0])[None].T
	kaggle_predictions = np.hstack((ids,predictions))
	writer = csv.writer(open(file, 'w'))
	if predictions.shape[1] == 1:
		writer.writerow(['# id','Prediction'])
	elif predictions.shape[1] == 2:
		writer.writerow(['# id','Prediction1', 'Prediction2'])

	writer.writerows(kaggle_predictions)

