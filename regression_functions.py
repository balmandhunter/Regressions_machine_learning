
# coding: utf-8

# In[3]:

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime as dt


from sklearn.cross_validation import cross_val_score 
from sklearn.cross_validation import KFold
import sklearn.preprocessing as pp
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Lasso
from sklearn.decomposition import RandomizedPCA
from sklearn.linear_model import Ridge
from sklearn import metrics
from sklearn import cross_validation
from sklearn.datasets import load_digits
from sklearn.learning_curve import learning_curve
import sklearn.decomposition
import sklearn.ensemble as sk
from sklearn import linear_model

import random
import sys
from scipy import stats


# In[ ]:




# In[ ]:

#Declare whether to process raw or filtered data.
def declare_filt_or_raw_dataset(which_data):
    if which_data == 0:
        ref_column = 'O3_ppb'
        leave_out_pod = 'pod_o3_smooth'
        pod_ozone = 'e2v03'
    else:
        ref_column = 'ref_o3_smooth'
        leave_out_pod = 'e2v03'
        pod_ozone = 'pod_o3_smooth'
    return ref_column, leave_out_pod, pod_ozone


# In[ ]:

####Define a function that makes numpy arrays out of the training and holdout data.


# In[ ]:

def make_numpy_arrays_for_tr_and_holdout(features, df_T, df_CV, ref_column):
    X_T = df_T[features].values
    X_CV = df_CV[features].values
    y_T = df_T[ref_column].values
    y_CV = df_CV[ref_column].values
    return X_T, y_T, X_CV, y_CV


# In[ ]:

###Add a 'day' column to the dataframe, and separate the data into training and holdout.


# In[2]:

def add_day_sep_tr_and_holdout(df, ref_column):
    #create a 'day' column in the dataframe by mapping the index column
    df['day'] = df.index.map(lambda dt: str(dt.month) + '-' + str(dt.day))
    days = df['day'].unique()
 
    
    num_days = np.argmax(days)
    days_tr = days[:num_days-1]
    df_tr = df.loc[df['day'] < days[num_days-1]]
    df_hold = df.loc[df['day'].isin([days[num_days], days[num_days-1]])]
    
    return df_tr, df_hold, days_tr


# In[1]:

#Scale the features and add a 'day' column to the dataframe.
def scale_features_and_create_day_column(df, ref_column):
    df_scaled = df.copy()
    #drop the day column from df_scaled
    df_scaled.drop('day', axis=1, inplace=True)
    
    features = list(df_scaled.ix[:,1:len(df.columns)])
    #Center feature values around zero and make them all have variance on the same order.
    df_scaled = df_scaled[features].apply(lambda x: pp.scale(x))
    df_sc = pd.concat([df_scaled, df[ref_column]], axis = 1)
    
    #add the 'day' column back in
    df_sc['day'] = df_sc.index.map(lambda dt: str(dt.month) + '-' + str(dt.day))
    
    return df_sc, features


# In[ ]:

#Define a custom cross-validation function.
def create_custom_cv(df):
    labels = df['day'].values
    lol = cross_validation.LeaveOneLabelOut(labels)


# In[ ]:

####Declare a neutral fitting function.
def fitting_func(model, X_T, y_T, X_CV, y_CV):    
    #fit a linear regression on the training data
    model.fit(X_T, y_T)   
    #find the normalized MSE for the training and holdout data
    return np.mean((y_CV - model.predict(X_CV))**2), np.mean((y_T - model.predict(X_T))**2)


# In[ ]:

####Define a function that loops through all of the days (CV by day), and computes MSE.
def cross_validation_by_day(model, features, df, days, ref_column):

    #initialize the holdout and training MSE
    day_date = []
    MSE_CV = [] 
    MSE_T = []
    #Calculate the training and holdout RSS for each step.
    #take the mean MSE for all of the possible holdout days (giving cross-validation error)
    for d in days:
        
        #call the df_subset function to make numpy arrays out of the training and holdout data
        X_T, y_T, X_CV, y_CV = make_numpy_arrays_for_tr_and_holdout(features, df[df.day != d], df[df.day == d], ref_column)
                
        MSE_CV_day, MSE_T_day = fitting_func(model, X_T, y_T, X_CV, y_CV)
         
        #record the MSE for lambda for the day
        MSE_CV.append(MSE_CV_day)
        MSE_T.append(MSE_T_day)
    
        #record the day
        day_date.append(d)
            
        #find the mean MSE of all of the days for the given value of lambda
        mean_CV_MSE_all_days = np.mean(MSE_CV)
        mean_train_MSE_all_Days = np.mean(MSE_T)
        
    print "Cross-Validation MSE: ", int(mean_CV_MSE_all_days), " Training MSE: ", int(mean_train_MSE_all_Days)

    return mean_CV_MSE_all_days, mean_train_MSE_all_Days 


# ###Plot Learning Curves

# In[ ]:

from sklearn.learning_curve import learning_curve

def plot_learning_curve(estimator, title, X, y, ylimit, cv, train_sizes, scoring):

    plt.figure()
    plt.title(title)
    if ylimit is not None:
        plt.ylim(ylimit)
    plt.xlabel("Training examples")
    plt.ylabel("Mean Squared Error")
    train_sizes, train_scores, valid_scores = learning_curve(estimator, X, y, cv = cv, train_sizes = train_sizes, scoring = scoring)
    train_scores_mean = -np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    valid_scores_mean = -np.mean(valid_scores, axis=1)
    valid_scores_std = np.std(valid_scores, axis=1)
    plt.grid(b=True, which='major', color='g', linestyle='-.')

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, valid_scores_mean - valid_scores_std, valid_scores_mean + valid_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training error")
    plt.plot(train_sizes, valid_scores_mean, 'o-', color="g", label="Cross-validation error")

    plt.legend(loc="best")
    return plt


# ####Define a function to find fitted values.

# In[ ]:

def find_fitted_cv_values_for_best_features(df, fs_features, num_good_feat, Model, days, ref_column):
    fitted_holdout_o3 = []
    for d in days:    
        #call the df_subset function to make numpy arrays out of the training and holdout data
        X_T, y_T, X_CV, y_CV = make_numpy_arrays_for_tr_and_holdout(fs_features[:num_good_feat], df[df.day != d], df[df.day == d], ref_column) 
        #fit a linear regression on the training data
        model = Model
        model.fit(X_T, y_T)
        
        if d == days[0]:
            fitted_CV_o3 = model.predict(X_CV)
        else:
            fitted_CV_o3 = np.concatenate((fitted_CV_o3, model.predict(X_CV)))

    df_lin_regr_best_feat = df.copy()
    df_lin_regr_best_feat['O3_fit'] = fitted_CV_o3
    return df_lin_regr_best_feat


# ###Define functions for plotting fitted vs. holdout data and residuals.

# In[4]:

def fitted_vs_ref_plot(df, i, ref_column):
    plt.figure(figsize = (5,5))
    plt.plot(df[ref_column],df.O3_fit,linestyle = '',marker = '.',alpha = 0.3)
    plt.xlabel('Reference O3 Conc.')
    plt.ylabel('Predicted O3 Conc (Cross-Validation)')
    plt.plot([1,df[ref_column].max()],[1,df[ref_column].max()])
    if i != 0:
        plt.title('Number of features = ' + str(i))


# ###Define a function that assigns time chunks to each pod for plotting

# In[ ]:

def assign_pod_calibration_times(pod_num, time_chunk):
    if time_chunk == 1:
        if pod_num == 'D0' or pod_num == 'F3' or pod_num == 'F4' or pod_num == 'D3' or pod_num == 'F5' or pod_num == 'F6'  or pod_num == 'F7':
            xlim = ['2014-07-11 00:00:00', '2014-07-13 00:00:00']
        elif pod_num == 'D8' :
            xlim = ['2014-07-11 00:00:00', '2014-7-12 00:00:00']
        elif pod_num == 'D4' or pod_num == 'D6' or pod_num == 'D8' or pod_num == 'N4' or pod_num == 'N7' or pod_num == 'N8':
            xlim = ['2014-07-13 00:00:00', '2014-7-15 00:00:00']
        elif pod_num == 'N3' or pod_mun == 'N5':
            xlim = ['2014-07-8 00:00:00', '2014-7-11 00:00:00']
    else: 
        if pod_num == 'D0':
            xlim = ['2014-08-30 00:00:00', '2014-09-4 00:00:00']
        elif pod_num == 'D4' or pod_num == 'F4':
            xlim = ['2014-08-15 00:00:00', '2014-08-21 00:00:00']
        elif pod_num == 'D3' or pod_num == 'D6' or pod_num == 'F3' or pod_num == 'D8' or pod_num == 'F5' or pod_num == 'F6' or pod_num == 'N8':
            xlim = ['2014-08-21 00:00:00', '2014-08-30 00:00:00']
        elif pod_num == 'F7' or pod_num == 'N4':
            xlim = ['2014-08-15 00:00:00', '2014-08-21 00:00:00']
        elif pod_num == 'N3':
            xlim = ['2014-08-14 00:00:00', '2014-08-21 00:00:00']
        elif pod_num == 'D4' or pod_num == 'N5':
            xlim = ['2014-08-29 00:00:00', '2014-09-4 00:00:00']
        elif pod_num == 'N7':
            xlim = ['2014-08-16 00:00:00', '2014-08-22 00:00:00']
    return xlim


# In[ ]:

def plot_fitted_and_ref_vs_time(df, pod_num, time_chunk, ref_column):
    plt.figure(figsize = (15,10))
    df[ref_column].plot(marker = '.',linestyle = ' ',)
    xlim = assign_pod_calibration_times(pod_num, time_chunk)
    df.O3_fit.plot(marker = '.',linestyle = ' ', xlim=xlim)
    plt.ylabel('Residual Value')


# In[ ]:

def plot_resid_vs_conc(df, ref_column):
    #find the residuals
    resid = df[ref_column] - df.O3_fit
    #plot the residuals to check for non-linearity of response predictor
    plt.figure(figsize = (15,5))
    plt.plot(df.O3_fit, resid, linestyle = '',marker = '.',alpha = 0.4)
    plt.plot([-40,70],[0,0], linestyle = ' ', marker = '.')
    plt.xlabel('Fitted O3 Conc.')
    plt.ylabel('Residuals')
    return resid


# In[ ]:

def plot_resid_vs_time(resid, pod_num, time_chunk):
    plt.figure(figsize = (15,5))
    xlim = assign_pod_calibration_times(pod_num, time_chunk)
    resid.plot(linestyle = '',marker = '.', xlim = xlim)
    #plt.plot([0,0],[70,0])
    plt.xlabel('Fitted O3 Conc.')
    plt.ylabel('Residuals')


# ###Make custom scoring function.

# In[5]:

from sklearn.metrics import make_scorer
def custom_mse_scoring_function(y, y_pred):
    low_sum = np.mean(0.1*(y[y < 60] - y_pred[y < 60])**2)
    high_sum = np.mean((y[y >= 60] - y_pred[y >= 60])**2)
    if np.isnan(low_sum) == True:
        low_sum = 0
    if np.isnan(high_sum) == True:
        high_sum = 0
    return int(low_sum + high_sum)

def custom_mae_scoring_function(y, y_pred):
    low_sum = np.mean(0.1*np.absolute(y[y < 65] - y_pred[y < 65]))
    high_sum = np.mean(np.absolute(y[y >= 65] - y_pred[y >= 65]))
    if np.isnan(low_sum) == True:
        low_sum = 0
    if np.isnan(high_sum) == True:
        high_sum = 0
    return int(low_sum + high_sum)


# ###Find the average score for all days.

# In[ ]:

def avg_cv_score_for_all_days(df, features, ref_column, model, scoring_metric,lol):
    X = df[features].values
    y = df[ref_column].values
    if scoring_metric == 'custom_mse':
        score_cv = -np.mean(cross_val_score(model, X, y, cv = lol, scoring = make_scorer(custom_mse_scoring_function, greater_is_better = False)))        
    elif scoring_metric == 'custom_mae':
        score_cv = -np.mean(cross_val_score(model, X, y, cv = lol, scoring = make_scorer(custom_mae_scoring_function, greater_is_better = False)))        
    else:
        score_cv = -np.mean(cross_val_score(model, X, y, cv = lol, scoring = scoring_metric))
    return score_cv


# In[ ]:

def forward_selection_step(model, b_f, features, df, ref_column, scoring_metric, lol):
    #initialize min_MSE with a very large number
    min_score = sys.maxint
    min_r2 = 0
    next_feature = ''

    for f in features:
        score_step = avg_cv_score_for_all_days(df, b_f + [f], ref_column, model, scoring_metric, lol)
        if score_step < min_score:
            min_score = score_step
            next_feature = f
            score_cv = "{:.1f}".format(min_score)   
    return next_feature, score_cv


# In[ ]:

def forward_selection_lodo(model, features, df, scoring_metric, ref_column, lol):
    #initialize the best_features list with the base features to force their inclusion
    best_features = ['days from start']
    #call the function that scales the features and creates a day column   
    
    score_cv = []
    MSE = []
    while len(features) > 0 and len(best_features) < 61:
        #next_features = []
        #score_cv_list = []
             
        next_feature, score_cv_feat = forward_selection_step(model, best_features, features, df, ref_column, scoring_metric, lol)
        #add the next feature to the list
        best_features += [next_feature]
        MSE.append("{:.1f}".format(-np.mean(cross_val_score(model, df[best_features].values, df[ref_column].values, cv = lol, scoring = 'mean_squared_error'))))
        score_cv.append(score_cv_feat)
        print 'Next best Feature: ', next_feature, ',', 'Score: ', score_cv_feat, ','
        
        #remove the added feature from the list
        features.remove(next_feature)
        
    print "Best Features: ", best_features
    return best_features, score_cv, MSE


# ###Plot the custom error and MSE as a function of number of features

# In[ ]:

def plot_error_vs_features(score, MSE):
    x = range(0, len(score))
    plt.plot(x, score, 'bo-')
    plt.plot(x, MSE, 'ro-')
    plt.ylim((0,100))
    plt.xlabel('Number of Features')
    plt.ylabel('Error')
    plt.grid(b=True, which='major', color='g', linestyle='-.')
    print 'Custom Score: ', score
    print 'MSE: ', MSE


# In[ ]:

def find_best_lambda(Model, features, df, ref_column, scoring_metric, cv, X, y):
    lambda_ridge = []
    mean_score_lambda = []
    i = 0.0000000001
    n = 1
    coefs = []
  

    while i < 100000000:
        #define the model
        model = Model(alpha=i)    
        #fit the ridge regression for the lambda
        model.fit(X, y)
        #record the custom score for this lambda value
        mean_score_lambda.append(avg_cv_score_for_all_days(df, features, ref_column, model, scoring_metric, cv))  
        #record the lambda value for this run
        lambda_ridge.append(i)
        #record the coefficients for this lambda value
        coefs.append(model.coef_)
        
        i = i*1.25
        n += 1  

    #find the lambda value (that produces the lowest cross-validation MSE)  
    best_lambda = lambda_ridge[mean_score_lambda.index(min(mean_score_lambda))]
    
    #record the MSE for this lambda value
    MSE = avg_cv_score_for_all_days(df, features, ref_column, Model(alpha=best_lambda), 'mean_squared_error', cv)
                                
    #plot the coefficients     
    ax = plt.gca()
    ax.set_color_cycle(['b', 'r', 'g', 'c', 'k', 'y', 'm'])

    ax.plot(lambda_ridge, coefs)
    ax.set_xscale('log')
    plt.xlabel('lambda')
    plt.ylabel('weights')
    plt.title(str(Model) + 'coefficients as a function of the regularization')
    plt.show()  
   
    #plot the results
    plt.plot(lambda_ridge, mean_score_lambda)
    ax = plt.gca()
    ax.set_xscale('log')
    plt.xlabel('lambda')
    plt.ylabel('Custom Score')
    
    print 'Best Lambda: ', best_lambda
    print 'Custom Error: ', int(min(mean_score_lambda))
    print 'CV Mean Squared Error: ', int(MSE)
    
    return best_lambda, min(mean_score_lambda), MSE 


# In[ ]:

def find_residuals_and_fitted_cv_values(Model, df, features, days, ref_column, best_lambda):
    model = Model(alpha = best_lambda)

    for d in days:               
        #call the function that defines the training and holdout data
        X_T, y_T, X_CV, y_CV = make_numpy_arrays_for_tr_and_holdout(features, df[df.day != d], df[df.day == d], ref_column)  
        #fit the ridge regression for the lambda
        model.fit(X_T, y_T)
        if d == days[0]:
            fitted_holdout_o3 = model.predict(X_CV)
        else:
            fitted_holdout_o3 = np.concatenate((fitted_holdout_o3, model.predict(X_CV)))
                
    df_ridge_fit = df.copy()
    df_ridge_fit['O3_fit'] = fitted_holdout_o3
    print "Coefficients: ", model.coef_
    return df_ridge_fit


# In[ ]:

#fit random forest and finds MSE
def fit_rfr_and_find_MSE(features, df_T, df_CV, d):
    
    #initialize the numpy array that will hold the test-mse data
    mse_array_test = np.zeros((i_max,j_max))
    mse_array_train = np.zeros((i_max,j_max))
    
    if options == 0:
        rfr = sk.RandomForestRegressor(n_estimators=100, oob_score = True, n_jobs = -1)
        forest = sk.RandomForestClassifier(n_estimators=100, random_state=0)
        
        #call the function that defines the trainig and holdout data
        X_T, y_T, X_CV, y_CV = make_numpy_arrays_for_tr_and_holdout(features, df_T, df_CV, ref_column)                
        
        #fit a linear regression on the training data
        rfr.fit(X_T, y_T)  
        
        #fit the holdout data for the day
        df_H['O3_fit'] = rfr.predict(X_CV)
        
        #plot the feature importances
        plot_importance(rfr, forest)
        #plot_ref_and_pod_ozone_for_each_day(df_fit[df_fit.day != d], df_fit[df_fit.day == d])
        #plot_temp_and_rh_for_each_day(df_fit[df_fit.day != d], df_fit[df_fit.day == d])
        #plot_fitted_and_ref_ozone_for_each_day(df_H['O3_fit'], df_fit[df_fit.day == d])
        
        MSE_CV = int(np.mean((y_CV - rfr.predict(X_CV))**2))
            
        print d,'Cross-Validatin MSE: ', MSE_CV
        return MSE_CV
        
    else:
        i_max = 1 # max features
        j_max = 1 # max depth
        i_min = 0
        j_min = 0
        #loop through all combinations of max_features and max_depth
        for i in range(i_min,i_max):
            j = j_min
            while j < j_max:
                #Set up the random forest regression features
                rfr = sk.RandomForestRegressor(n_estimators=250, oob_score = True, n_jobs = -1, max_features = i+1, max_depth = j+1)
                forest = sk.RandomForestClassifier(n_estimators=100, random_state=0)
                        
                #call the r=function that defines the trainig and holdout data
                X_T, y_T, X_CV, y_CV = make_numpy_arrays_for_tr_and_holdout(features, df_T, df_CV, ref_column)   
                
                #fit a linear regression on the training data
                rfr.fit(X_T, y_T)  
                #plot_importance(rfr, forest)
            
                #add the mse for each i and j to the 2D array (i is on one axis, j is on the other, and mse is a grid)
                mse_array_CV[i,j] = int(np.mean((y_CV - rfr.predict(X_CV))**2))
            
                j += 10
               
        #find the MSE for the training and holdout data
        return mse_array_CV


# In[ ]:

def plot_importance(rfr,forest):
    importances = rfr.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    print std
    indices = np.argsort(importances)[::-1]
    
    # Print the feature ranking
    print("Feature ranking:")
    for f in range(len(fs_features)):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]])),fs_features[indices[f]]
    
    #Plot the feature importances of the forest
    plt.figure(figsize=(15,5))
    plt.title("Feature importances")
    plt.bar(range(len(fs_features)), importances[indices], color="r", align="center")
    #, yerr = std[indices]
    plt.xticks(range(len(fs_features)), indices)
    plt.xlim([-1, len(fs_features)])
    plt.show()
    

# In[ ]:

def find_MSE_random_forest(df, features, days):
    count = 1
    #Calculate the training and holdout RSS for each step.
    #take the mean MSE for all of the possible holdout days (giving cross-validation error)
    for d in days:
        MSE_CV_day = fit_rfr_and_find_MSE(fs_features, df_fits[df_fits.day != d], df_fits[df_fits.day == d], d)

        day_date.append(d)
        #count_append.append(count)

        if count == 1 and options == 1:
            MSE_CV = MSE_CV_day
        elif count == 1:
            MSE_CV.append(MSE_CV_day)
        elif options == 1:
            MSE_CV = np.dstack((MSE_CV,MSE_CV_day))
        else:
            MSE_CV.append(MSE_CV_day)

        count +=1   
    return MSE_CV

    
# In[ ]:

def plot_temp_and_rh_for_each_day(df_T, df_H):
    fig = plt.figure(figsize=(15,5))
    ax = fig.add_subplot(111)
    plt.title('Temp and Rh Data', fontsize = 30)
    ax.plot(df_H['Temp'],  color="r", marker = '.', linestyle = '--', label = 'reference')
    ax.set_xlabel('Time', fontsize = 18)
    ax.set_ylabel('Temperature (as % of maximum)', fontsize = 18)
    ax.legend()
    
    ax2 = ax.twinx()  
    ax2.set_ylabel('Rel. Hum. (as % of maximum)', fontsize = 18)
    ax2.legend(loc = 0)
    plt.plot((df_H['Rh']), marker = '.', linestyle = '--', label = 'pod')
    plt.show()



# In[ ]:

#__all__ = ["echo", "surround", "reverse"]

def plot_ref_and_pod_ozone_for_each_day(df_T, df_H):
    fig = plt.figure(figsize=(15,5))
    ax = fig.add_subplot(111)
    plt.title('Pod and Reference Ozone Data', fontsize = 30)
    ax.plot(df_H['O3_ppb'], color="r", marker = '.', linestyle = '--', label = 'reference')
    ax.set_xlabel('Time', fontsize = 18)
    ax.set_ylabel('Reference Ozone', fontsize = 18)
    ax.legend()
    
    
    df_H['ones'] = 1
    df_H['inverse_o3'] = df_H['ones'].div(df_H['e2v03'], axis='index')
    ax2 = ax.twinx()  
    ax2.set_ylabel('Pod Ozone (1/mV)', fontsize = 18)
    ax2.legend(loc = 0)
    plt.plot((df_H['inverse_o3']), marker = '.', linestyle = '--', label = 'pod')
    plt.show()
    


# In[ ]:

def plot_fitted_and_ref_ozone_for_each_day(fitted_data, df_H):
    plt.figure(figsize=(15,5))
    plt.title('Fitted and Ref. Ozone Data', fontsize = 30)
    fitted_data.plot(color="r", marker = '.', label = 'fitted')
    plt.xlabel('Time', fontsize = 18)
    plt.ylabel('Ozone (ppb)', fontsize = 18)
    plt.legend() 
    df_H['O3_ppb'].plot(label = 'reference')
    plt.show()


# In[ ]:

def find_daily_min_max(features, df_T, df_H,d):
    X_T = df_T[features]
    X_H = df_H[features]
    y_T = df_T['O3_ppb']
    y_H = df_H['O3_ppb']
    return y_H.max(), df_H['Temp'].max(), df_H['Rh'].max(), y_H.min(), df_H['Temp'].min(), df_H['Rh'].min(), y_H.mean(), df_H['Temp'].mean(), df_H['Rh'].mean(), y_H.std(), df_H['Temp'].std(), df_H['Rh'].std(), df_H['e2v03'].max(), df_H['e2v03'].min(), df_H['e2v03'].mean(), df_H['e2v03'].std()


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



if __name__ == "__main__":
    import sys
    fib(int(sys.argv[1]))


