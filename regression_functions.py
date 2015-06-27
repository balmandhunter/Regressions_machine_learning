import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime as dt
import pylab as pl
import random
import sys
from scipy import stats

import sklearn.ensemble as sk
from sklearn import linear_model
from sklearn import metrics
from sklearn import cross_validation
from sklearn.cross_validation import cross_val_score 
from sklearn.metrics import make_scorer
from sklearn.cross_validation import KFold
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Lasso
from sklearn.decomposition import RandomizedPCA
from sklearn.linear_model import Ridge
from sklearn.datasets import load_digits
from sklearn.learning_curve import learning_curve
import sklearn.decomposition
import sklearn.preprocessing as pp


#Declare whether to process raw or filtered data.
def declare_filt_or_raw_dataset(which_data):
    if which_data == 0:
        ref_column = 'O3_ppb'
        leave_out_pod = 'pod_o3_smooth'
        pod_ozone = 'e2v03'
    else:
        ref_column = 'ref_o3_smooth'
        leave_out_pod = 'pod_o3_smooth'
        pod_ozone = 'e2v03'
    return ref_column, leave_out_pod, pod_ozone

def sci_minmax(X):
    minmax_scale = pp.MinMaxScaler(feature_range=(0, 1), copy=True)
    return minmax_scale.fit_transform(X)


def scale_features_and_create_day_column(df, ref_column):
    df_prescaled = df.copy().astype(float)
    df_scaled = df_prescaled.copy() 
    features = [x for x in list(df_scaled.ix[:,0:len(df.columns)]) if x != ref_column]
    #prescale the features
    df_prescaled = df_scaled[features].apply(lambda x: sci_minmax(x))
    #Center feature values around zero and make them all have variance on the same order.
    df_scaled = df_prescaled[features].apply(lambda x: pp.scale(x))
    df_sc = pd.concat([df_scaled, df[ref_column]], axis = 1)  
    #add a 'day' column 
    df_sc['day'] = df_sc.index.map(lambda dt: str(dt.month) + '-' + str(dt.day))
    #add a column that has the day and 'AM' or "PM"
    df_sc['chunk'] = df_sc.index.map(lambda dt: str(dt.month) + '-' + str(dt.day) + " " + ("AM" if dt.hour < 12 else "PM"))
    return df_sc, features


def sep_tr_and_holdout(df, ref_column):
    #find the unique values of the day + AM/PM column
    chunk_list = df.chunk.unique()
    #shuffle the chunks of time
    np.random.shuffle(chunk_list)
    #declare the first 4 chunks of the randomized list to be the holdout chunks
    hold_chunks = chunk_list[0:4]
    chunks_tr = chunk_list[4:]
    df_tr = df[~df.chunk.isin(hold_chunks)]
    days_tr = df_tr['day'].unique()
    df_hold = df[df.chunk.isin(hold_chunks)]
    #df_hold = df_hold[:len(df_hold['day'])-90]
    return df_tr, df_hold, chunks_tr, days_tr


def create_custom_cv(df):
    labels = df['chunk'].unique()
    lol = cross_validation.LeaveOneLabelOut(labels)


def numpy_arrays_for_tr_and_cv(features, df_T, df_CV, ref_column):
    X_T = df_T[features].values
    X_CV = df_CV[features].values
    y_T = df_T[ref_column].values
    y_CV = df_CV[ref_column].values
    return X_T, y_T, X_CV, y_CV

def daily_numpy_arrays_for_tr_and_cv(features, df_tr, ref_column, days_tr):
    first = True
    for d in days_tr:
        df_T = df_tr[df_tr['day'] != d]
        df_T = df_tr[df_tr['day'] == d]
        features = [f for f in features if f not in 'leave_out']
        X_T = df_tr[features].values
        X_CV = df_tr[features].values
        y_T = df_tr[ref_column].values
        y_CV = df_tr[ref_column].values
        if first:
            X_T_matrix = [X_T]
            X_CV_matrix = [X_CV]
            y_T_matrix = [y_T]
            y_CV_matrix = [y_CV]
            first = False
        else:
            X_T_matrix.append(X_T)
            X_CV_matrix.append(X_CV)
            y_T_matrix.append(y_T)
            y_CV_matrix.append(y_CV)
    return X_T_matrix, X_CV_matrix, y_T_matrix, y_CV_matrix


def numpy_arrays_for_holdout_and_training(features, df_H, df_tr, ref_column):
    features = [f for f in features if f not in 'chunk']
    X_T = df_tr[features].values
    X_H = df_H[features].values
    y_T = df_tr[ref_column].values
    y_H = df_H[ref_column].values
    return X_H, y_H, X_T, y_T


def fitting_func(model, X_T, y_T, X_CV, y_CV):    
    #fit a linear regression on the training data
    model.fit(X_T, y_T)   
    #find the normalized MSE for the training and holdout data
    score_cv_day = np.mean((y_CV[y_CV >= 60] - model.predict(X_CV)[y_CV >= 60])**2)
    return np.mean((y_CV - model.predict(X_CV))**2), np.mean((y_T - model.predict(X_T))**2), model.predict(X_CV), score_cv_day


def find_predicted_holdout_data(df_H, features, df_tr, ref_column, model):
    df_H = df_H.copy()
    X_H, y_H, X_T, y_T = numpy_arrays_for_holdout_and_training(features, df_H, df_tr, ref_column)
    model.fit(X_T, y_T)
    df_H['O3_fit'] = model.predict(X_H)
    df_H['ref_fit'] = y_H
    #find the t_stat anf p_value
    t_stat, p_value = stats.ttest_ind(model.predict(X_H), y_H, equal_var = False)
    #find the difference in means between the high reference and predicted data
    diff_in_mean = np.mean(model.predict(X_H)[y_H >= 60]) - np.mean(y_H[y_H >= 60])
    MSE_H = np.mean((y_H - model.predict(X_H))**2)
    MSE_H_high = np.mean((y_H[y_H >= 60] - model.predict(X_H)[y_H >= 60])**2)

    return df_H, MSE_H, MSE_H_high, t_stat, p_value, round(diff_in_mean, 1)


def find_predicted_cv_data(df_tr, X_pred_cv_all, y_CV_all):
    df_cv = df_tr.copy()
    df_cv['O3_fit'] = X_pred_cv_all
    df_cv['ref_fit'] = y_CV_all
    return df_cv


def print_stats(train_MSE, CV_MSE, score_cv, diff_in_mean_cv, MSE_H, score_H, diff_in_mean_H, diff_in_mean_cv_highest):
    print "Training RMSE:", round(np.sqrt(train_MSE),1)
    print (
        "Cross-Validation RMSE: " + str(round(np.sqrt(CV_MSE))) + " , " +
        "High-Value CV RMSE: " + str(round(np.sqrt(score_cv))) + " , " 
        )
    print (
        "CV High Diff. in Mean (>60): " + str(round(diff_in_mean_cv, 1)) + " , " +
        "CV High Diff. in Mean (>70): " + str(round(diff_in_mean_cv_highest, 1)) 
        )

    print (
        "Holdout RMSE: " + str(round(np.sqrt(MSE_H))) +  " , " +  
        "High-Value Holdout RMSE: " + str(round(np.sqrt(score_H))) + " , "
        "Holdout High Diff. in Mean.: " + str(diff_in_mean_H) 
        )
    

#Define a function that loops through all of the days (CV by day), and computes MSE.
def cross_validation_by_day(model, features, df_tr, df_H, days, ref_column):
    #initialize the holdout and training MSE
    MSE_CV = np.zeros(len(days)) 
    MSE_T = np.zeros(len(days)) 
    score_cv_all = np.zeros(len(days))
    count = 0
    y_CV_all = []
    X_pred_cv_all = []
    first = True
    #Calculate the training and holdout RSS for each step.
    #take the mean MSE for all of the possible holdout days (giving cross-validation error)
    for d in days:   
        #call the df_subset function to make numpy arrays out of the training and holdout data
        X_T, y_T, X_CV, y_CV = numpy_arrays_for_tr_and_cv(
            features, 
            df_tr[df_tr.day != d], 
            df_tr[df_tr.day == d], 
            ref_column
        )   
        MSE_CV_day, MSE_T_day, X_pred_cv, score_cv_day = fitting_func(model, X_T, y_T, X_CV, y_CV)
        #record the MSE for lambda for the day
        MSE_CV[count] = MSE_CV_day
        MSE_T[count] = MSE_T_day
        if not np.isnan(score_cv_day):
            score_cv_all[count] = score_cv_day

        if first:
            X_pred_cv_all = X_pred_cv
            y_CV_all = y_CV
            first = False
        else:
            X_pred_cv_all = np.concatenate([X_pred_cv_all, X_pred_cv])
            y_CV_all = np.concatenate([y_CV_all, y_CV])
        count += 1
    #remove the zeros from the high-value score (the zeros are from data chunks where ozone conc. never passed the high limit)
    score_cv_all =  filter(lambda a: a != 0, score_cv_all)    
    score_cv = np.mean(score_cv_all)        
    #find the mean MSE of all of the days for the given value of lambda
    mean_CV_MSE_all_days = np.mean(MSE_CV)
    mean_train_MSE_all_Days = np.mean(MSE_T)
    #find the predicted values for the cross-validation data and put them in a dataframe
    df_cv = find_predicted_cv_data(df_tr, X_pred_cv_all, y_CV_all)
    #find the predicted values for the holdout data and put them in a dataframe
    df_H, MSE_H, score_H, t_stat, p_value, diff_in_mean_H = find_predicted_holdout_data(df_H, features, df_tr, ref_column, model)
    #find the percentage difference between the high reference and predicted values
    diff_in_mean_cv = np.mean(X_pred_cv_all[y_CV_all >= 60]) - np.mean(y_CV_all[y_CV_all >= 60])
    diff_in_mean_cv_highest = np.mean(X_pred_cv_all[y_CV_all >= 70]) - np.mean(y_CV_all[y_CV_all >= 70])
    #print out important stats
    print_stats(mean_train_MSE_all_Days, mean_CV_MSE_all_days, score_cv, diff_in_mean_cv, MSE_H, score_H, diff_in_mean_H, diff_in_mean_cv_highest) 
    return mean_CV_MSE_all_days, mean_train_MSE_all_Days, MSE_H, score_cv, X_pred_cv_all, y_CV_all, df_cv, df_H


def find_fitted_cv_values_for_best_features(df_T, df_H, fs_features, num_good_feat, Model, chunk, ref_column):
    fitted_holdout_o3 = []
    first = True
    for d in chunk:    
        #call the df_subset function to make numpy arrays out of the training and holdout data
        X_T, y_T, X_CV, y_CV =  numpy_arrays_for_tr_and_cv(fs_features[:num_good_feat], 
            df_T[df_T.chunk != d], df_T[df_T.chunk == d], ref_column) 
        #fit a linear regression on the training data
        model = Model
        model.fit(X_T, y_T)
        
        if first:
            fitted_CV_o3 = model.predict(X_CV)
            y_CV_all = y_CV
            first = False
        else:
            fitted_CV_o3 = np.concatenate((fitted_CV_o3, model.predict(X_CV)))
            y_CV_all = np.concatenate([y_CV_all, y_CV])

    df_cv = df_T.copy()
    df_cv['O3_fit'] = fitted_CV_o3
    df_cv['ref_fit'] = y_CV_all

    df_H = df_H.copy()
    X_H, y_H, X_T, y_T = numpy_arrays_for_holdout_and_training(fs_features, df_H, df_T, ref_column)
    model.fit(X_T, y_T)
    df_H['O3_fit'] = model.predict(X_H)
    return df_cv, df_H


def custom_high_scoring_function(y, y_pred):
    high_sum = np.mean(((y - y_pred)[y >= 65])**2)
    return high_sum


def custom_mse(y, y_pred):
    return np.mean( (y - y_pred)**2 )


def custom_mse_scoring_function(y, y_pred):
    low_MSE = np.nan_to_num(np.mean( 0.1*((y - y_pred)[y < 60])**2 ))
    high_MSE = np.nan_to_num( np.mean( ( (y - y_pred)[y >= 60] )**2 ) )
    diff_in_median_cv = np.nan_to_num((np.median(y_pred[y >= 60]) - np.median(y[y >= 60]))) 

    return (np.sqrt((low_MSE + high_MSE))*2 + diff_in_median_cv)


def avg_cv_score_for_all_days(df, features, ref_column, model, scoring_metric, days_tr):
    first = True
    custom_score = np.zeros(len(days_tr))
    count = 0
    for d in days_tr:  
        #call the df_subset function to make numpy arrays out of the training and holdout data
        X_T, y_T, X_CV, y_CV = numpy_arrays_for_tr_and_cv(features, df[df.day != d], df[df.day == d], ref_column)   
        lin_regr = model.fit(X_T, y_T)        
        y_pred_day = lin_regr.predict(X_CV)
        custom_score_day = custom_mse_scoring_function(y_CV, y_pred_day)
        custom_score[count] = custom_score_day
        count += 1

    #remove the zeros from the high-value score (the zeros are from days where ozone conc. never passed the high limit)
    custom_score_all =  filter(lambda a: a != 0, custom_score)    
    score_cv = round(np.mean(custom_score_all), 1)     
    return score_cv


def forward_selection_step(model, b_f, features, df, ref_column, scoring_metric, days_tr):
    #initialize min_MSE with a very large number
    min_score = 100000000
    next_feature = ''
    for f in features:
        score_cv_step = avg_cv_score_for_all_days(df, b_f + [f], ref_column, model, scoring_metric, days_tr)
        if score_cv_step < min_score:
            min_score = score_cv_step
            next_feature = f
            score_cv = round(min_score, 1)
    return next_feature, score_cv


def forward_selection_lodo(model, features, df, scoring_metric, ref_column, days_tr, n_feat):
    #initialize the best_features list with the base features to force their inclusion
    best_features = []
    score_cv = []
    RMSE = []
    while len(features) > 0 and len(best_features) < n_feat:   
        next_feature, score_cv_feat = forward_selection_step(model, best_features, features, df, ref_column, scoring_metric, days_tr)
        #add the next feature to the list
        best_features += [next_feature]
        MSE_feat = -np.mean(cross_val_score(model, df[best_features].values, df[ref_column].values, 
            cv = cross_validation.LeaveOneLabelOut(days_tr), scoring = 'mean_squared_error'))
        RMSE_features = round(np.sqrt(MSE_feat), 1)
        score_cv.append((score_cv_feat))
        RMSE.append(RMSE_features)
        print 'Next best Feature: ', next_feature, ',', 'Score: ', score_cv_feat, 'RMSE: ', RMSE_features, "#:", len(best_features)
        #remove the added feature from the list
        features.remove(next_feature)    
    print "Best Features: ", best_features
    return best_features, score_cv, RMSE


def avg_cv_score_for_all_days_ridge(df, features, ref_column, model, scoring_metric, cv):
    X = df[features].values
    y = df[ref_column].values
    score_cv = -np.mean(cross_val_score(model, X, y, cv = cv, 
        scoring = make_scorer(scoring_metric, greater_is_better = False)))        
    return score_cv


def find_best_lambda(Model, features, df, ref_column, scoring_metric, days_tr, X, y):
    lambda_ridge = []
    mean_score_lambda = []
    i = 0.000001
    n = 1
    coefs = []
    while i < 10000:
        print i
        #define the model
        model = Model(alpha=i)    
        #fit the ridge regression for the lambda
        model.fit(X, y)
        #record the custom score for this lambda value
        mean_score_lambda.append(avg_cv_score_for_all_days(df, features, ref_column, model, scoring_metric, days_tr))  
        #record the lambda value for this run
        lambda_ridge.append(i)
        #record the coefficients for this lambda value
        coefs.append(model.coef_)
        i = i * 2
        n += 1 

    #find the lambda value (that produces the lowest cross-validation MSE)  
    best_lambda = lambda_ridge[mean_score_lambda.index(min(mean_score_lambda))] 
    model = Model(alpha = best_lambda)   
    #record the MSE for this lambda value
    MSE = avg_cv_score_for_all_days(df, features, ref_column, Model(alpha=best_lambda), 'mean_squared_error', days_tr)   
    
    print 'Best Lambda:', best_lambda, ",", 'CV RMSE:', round(np.sqrt(MSE),1), "," , 'High-Value RMSE:', round(np.sqrt(min(mean_score_lambda)),1) 
    return best_lambda, min(mean_score_lambda), MSE, lambda_ridge, coefs, mean_score_lambda, Model


def find_residuals_and_fitted_cv_values(Model, df, features, days, ref_column, best_lambda):
    model = Model(alpha = best_lambda)
    first = True
    for d in days:               
        #call the function that defines the training and holdout data
        X_T, y_T, X_CV, y_CV = numpy_arrays_for_tr_and_cv(features, df[df.day != d], df[df.day == d], ref_column)  
        #fit the ridge regression for the lambda
        model.fit(X_T, y_T)
        if first:
            fitted_holdout_o3 = model.predict(X_CV)
            y_CV_all = y_CV
            first = False
        else:
            fitted_holdout_o3 = np.concatenate((fitted_holdout_o3, model.predict(X_CV)))
            y_CV_all = np.concatenate((y_CV_all, y_CV))
                
    df_ridge_fit = df.copy()
    df_ridge_fit['O3_fit'] = fitted_holdout_o3
    df_ridge_fit['ref_fit'] = y_CV_all
    return df_ridge_fit


#fit random forest and finds MSE
def fit_rfr_and_find_MSE(features, df_T, df_CV, d, options, ref_column):
    
    if options == 0:
        rfr = sk.RandomForestRegressor(n_estimators=10, oob_score = True, n_jobs = -1)
        forest = sk.RandomForestClassifier(n_estimators=10, random_state=0)
        #call the function that defines the training and holdout data
        X_T, y_T, X_CV, y_CV = numpy_arrays_for_tr_and_cv(features, df_T, df_CV, ref_column)                
        #fit a linear regression on the training data
        rfr.fit(X_T, y_T)  
        #fit the holdout data for the day
        df_CV_rf = df_CV.copy()
        df_CV_rf['O3_fit'] = rfr.predict(X_CV)
        #plot the feature importances
        plot_importance(rfr, forest, features)
        MSE_CV = int(np.mean((y_CV - rfr.predict(X_CV))**2))
            
        print d,'Cross-Validatin MSE: ', MSE_CV
        return MSE_CV, df_CV_rf
        
    else:
        i_max = 10 # max features
        j_max = 10 # max depth
        i_min = 0
        j_min = 0
         #initialize the numpy array that will hold the test-mse data
        mse_array_CV = np.zeros((i_max,j_max))
        #loop through all combinations of max_features and max_depth
        for i in range(i_min,i_max):
            j = j_min
            while j < j_max:
                #Set up the random forest regression features
                rfr = sk.RandomForestRegressor(n_estimators=10, oob_score = True, n_jobs = -1, 
                    max_features = i+1, max_depth = j+1)
                forest = sk.RandomForestClassifier(n_estimators=10, random_state=0)      
                #call the function that defines the trainig and holdout data
                X_T, y_T, X_CV, y_CV =  numpy_arrays_for_tr_and_cv(features, df_T, df_CV, ref_column)   
                #fit a linear regression on the training data
                rfr.fit(X_T, y_T)  
                #add the mse for each i and j to the 2D array (i is on one axis, j is on the other, and mse is a grid)
                mse_array_CV[i,j] = int(np.mean((y_CV - rfr.predict(X_CV))**2))
            
                j += 10
               
        #find the MSE for the training and holdout data
        return mse_array_CV


def find_MSE_random_forest(df, features, chunk, options, ref_column):
    MSE_CV = []
    count = 1
    #Calculate the training and holdout RSS for each step.
    #take the mean MSE for all of the possible holdout days (giving cross-validation error)
    for d in chunk:
        if options == 0:
            MSE_CV_day, df_rf_CV = fit_rfr_and_find_MSE(features, df[df.chunk != d], df[df.chunk == d], d, options, ref_column)
        else: 
            MSE_CV_day = fit_rfr_and_find_MSE(features, df[df.chunk != d], df[df.chunk == d], d, options, ref_column)
        
        if count == 1 and options == 0:
            df_rf = df_rf_CV
        elif options == 0:
            df_rf = pd.concat([df_rf, df_rf_CV])

        if count == 1:
            MSE_CV = MSE_CV_day
        else:
            #MSE_CV.append(MSE_CV_day)
            MSE_CV = np.dstack((MSE_CV,MSE_CV_day))
        count +=1   
    if options == 0:
        return MSE_CV, df_rf
    else:
        return MSE_CV
    

def find_daily_min_max(features, df_T, df_H,d):
    X_T = df_T[features]
    X_H = df_H[features]
    y_T = df_T['O3_ppb']
    y_H = df_H['O3_ppb']
    return y_H.max(), df_H['Temp'].max(), df_H['Rh'].max(), y_H.min(), df_H['Temp'].min(), df_H['Rh'].min(), 
    y_H.mean(), df_H['Temp'].mean(), df_H['Rh'].mean(), y_H.std(), df_H['Temp'].std(), df_H['Rh'].std(), 
    df_H['e2v03'].max(), df_H['e2v03'].min(), df_H['e2v03'].mean(), df_H['e2v03'].std()



if __name__ == "__main__":
    import sys
    fib(int(sys.argv[1]))


