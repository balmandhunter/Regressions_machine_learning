import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime as dt
import pylab as pl


def plot_params():
    a = plt.rc('xtick', labelsize = 20)
    b = plt.rc('ytick', labelsize = 20)
    figsize = (5,5)
    facecolor = 'w'
    size = 20
    return a, b, plt.gca(), figsize , facecolor, size


def fitted_vs_ref_plot(df, i, ref_column):
    a, b, axes, fig_size, face_color, label_size = plot_params()
    plt.figure(fig_params)
    plt.plot(df.ref_fit, df.O3_fit, linestyle = '', marker = '.', alpha = 0.3)
    plt.xlabel('Reference O3 Conc.', size = 12)
    plt.ylabel('Predicted O3 Conc (Cross-Validation)', size = 20)
    plt.plot([1, df.ref_fit.max()], [1,df.ref_fit.max()])
    axes.set_ylim([-20,100])
    if i != 0:
        plt.title('Number of features = ' + str(i))


def plot_fitted_and_ref_vs_time(df, pod_num, time_chunk, ref_column):
    a, b, axes, fig_params, label_size = plot_params()
    plt.figure(fig_size = (5,5))
    df.ref_fit.plot(marker = '.',linestyle = ' ')
    if time_chunk != 0:
        xlim = assign_pod_calibration_times(pod_num, time_chunk)
        df.O3_fit.plot(marker = '.',linestyle = ' ', xlim = xlim)
    else:
        df.O3_fit.plot(marker = '.',linestyle = ' ')
    axes = plt.gca()
    axes.set_ylim([-20,100])
    plt.legend()
    plt.ylabel('Ozone Concentration (ppb)')


def plot_hist(values, other, title):
    h = sorted(values)  
    plt.figure(figsize=(15,5))
    fit = stats.norm.pdf(h, np.mean(h), np.std(h))  #this is a fitting indeed
    pl.title(title)
    pl.plot(h, fit, '-o')
    abs_min_dec = min(min(values), min(other))
    abs_max_dec = max(max(values), max(other))
    abs_min = myround(abs_min_dec, 5)
    abs_max = myround(abs_max_dec, 5)
    pl.hist(h, normed=True, bins=np.arange(abs_min-10,abs_max+10, 5))  
        #use this to draw histogram of your data
    axes = plt.gca()
    axes.set_xlim([-40, 100])
    pl.show()  


def plot_error_vs_features(score, MSE):
    plt.rc('xtick', labelsize = 20) 
    plt.rc('ytick', labelsize = 20) 
    x = range(0, len(score))
    plt.plot(x, score, marker = '.', markersize = 20, label='Cust. Score')
    plt.plot(x, MSE, marker = '.', markersize = 20, label='MSE')
    axes = plt.gca()
    axes.set_ylim([0,60])
    plt.xlabel('Number of Features', size = 20)
    plt.ylabel('Error', size = 20)
    plt.grid(b=True, which='major', color='g', linestyle='-.')
    plt.legend(size = 14)

    print 'Custom Score: ', score
    print 'MSE: ', MSE


def plot_learning_curve(estimator, title, X, y, ylimit, cv, train_sizes, scoring):
    plt.figure(facecolor='w', figsize = (5,5), frameon = "True")
    plt.title(title, size = 12)
    plt.rc('xtick', labelsize = 20) 
    plt.rc('ytick', labelsize = 20) 
    plt.rc('font', **font)  # pass in the font dict as kwargs
    if ylimit is not None:
        axes = plt.gca()
        axes.set_ylim(ylimit)
    plt.xlabel("Training Samples", size = 20)
    plt.ylabel("Mean Squared Error", size = 20)
    train_sizes, train_scores, valid_scores = learning_curve(estimator, X, y, 
        cv = cv, train_sizes = train_sizes, scoring = scoring)
    train_scores_mean = -np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    valid_scores_mean = -np.mean(valid_scores, axis=1)
    valid_scores_std = np.std(valid_scores, axis=1)
    plt.grid(b=True, which='major', color='#696969', linestyle=':')
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, 
        alpha=0.1, color="r")
    plt.fill_between(train_sizes, valid_scores_mean - valid_scores_std, valid_scores_mean + valid_scores_std, 
        alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training error")
    plt.plot(train_sizes, valid_scores_mean, 'o-', color="g", label="Cross-validation error")
    leg = plt.legend(loc="best", prop={'size':14}, frameon = 'True')
    leg.get_frame().set_facecolor('w')
    #fig.savefig('learning_curve.png', bbox_inches= 'tight')
    return plt


def plot_resid_vs_conc(df, ref_column):
    #find the residuals
    resid = df.ref_fit - df.O3_fit
    #plot the residuals to check for non-linearity of response predictor
    plt.figure(figsize = (15,5))
    plt.plot(df.O3_fit, resid, linestyle = '',marker = '.',alpha = 0.4)
    plt.plot([-40,70],[0,0], linestyle = ' ', marker = '.')
    axes = plt.gca()
    axes.set_ylim([-80,80])
    axes.set_xlim([-20,100])
    plt.xlabel('Fitted O3 Conc.')
    plt.ylabel('Residuals')
    return resid


def plot_resid_vs_time(resid, pod_num, time_chunk):
    plt.figure(figsize = (15,5))
    xlim = assign_pod_calibration_times(pod_num, time_chunk)
    resid.plot(linestyle = '',marker = '.', xlim = xlim)
    #plt.plot([0,0],[70,0])
    plt.xlabel('Fitted O3 Conc.')
    plt.ylabel('Residuals')


def plot_lambda(lambda_ridge, coefs, mean_score_lambda, Model):
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


def plot_importance(rfr,forest, features):
    importances = rfr.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
    print std
    indices = np.argsort(importances)[::-1] 
    # Print the feature ranking
    print("Feature ranking:")
    for f in range(10):
        print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]])),features[indices[f]]  
    #Plot the feature importances of the forest
    plt.figure(figsize=(15,5))
    plt.title("Feature importances")
    plt.bar(range(10), importances[indices], color="r", align="center")
    #, yerr = std[indices]
    plt.xticks(range(len(features)), indices)
    plt.xlim([-1, len(features)])
    plt.show()
    

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
  

def plot_param_select_MSE(MSE_CV_per_day, i, j):  
    fig = plt.figure(figsize=(20, 20))

    ax = fig.add_subplot(111)
    ax.set_title('colorMap')
    imgplot = plt.imshow(MSE_CV_per_day)
    imgplot.set_cmap('hot')
    #imgplot.set_clim(60,71)
    ax.set_aspect('equal')

    plt.colorbar(orientation='vertical')
    plt.show()
    plt.xlabel('Maximum Tree Depth')
    plt.ylabel('Maximum Features at Each Split')
    
    i,j = np.where(MSE_CV_per_day == MSE_CV_per_day.min())
    print 'Max features = ' + str(i)
    print 'Max depth = ' + str(j)
    print 'MSE for the holdout data = ' + str(min_MSE_CV)


def plot_fitted_and_ref_ozone_for_each_day(fitted_data, df_H):
    plt.figure(figsize=(15,5))
    plt.title('Fitted and Ref. Ozone Data', fontsize = 30)
    fitted_data.plot(color="r", marker = '.', label = 'fitted')
    plt.xlabel('Time', fontsize = 18)
    plt.ylabel('Ozone (ppb)', fontsize = 18)
    plt.legend() 
    df_H['O3_ppb'].plot(label = 'reference')
    plt.show()


def plot_daily_mse_and_features_for_day(MSE_H, day_date,feat_to_compare, title, sec_axis_label):
    from matplotlib import rc
    rc('mathtext', default='regular')
    indices = day_date
    #Plot the feature importances of the forest
    fig = plt.figure(figsize=(15,5))
    ax = fig.add_subplot(111)
    plt.title(title, fontsize = 30)
    ax.bar(range(len(day_date)), MSE_H,  color="r", align="center")
    plt.xticks(range(len(day_date)), indices)
    plt.xlim([-1, len(day_date)])
    ax.set_xlabel('Date', fontsize = 18)
    ax.set_ylabel('MSE (ppb)', fontsize = 18)
    ax2 = ax.twinx()  
    ax2.set_ylabel(sec_axis_label, fontsize = 18)
    plt.plot(range(len(day_date)), feat_to_compare, marker = 'o', linestyle = '--')
    plt.show()


if __name__ == "__main__":
    import sys
    fib(int(sys.argv[1]))
