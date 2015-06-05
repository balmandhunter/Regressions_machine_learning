
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


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:

#__all__ = ["echo", "surround", "reverse"]


# In[ ]:

if __name__ == "__main__":
    import sys
    fib(int(sys.argv[1]))


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



