import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def find_lag_slope(df_P, int_min, data_col):
    df_P['diff'] = df_P[data_col].diff()
    
    slope = []
    for i in range(int_min,len(df_P['diff'])):
        top = i-int_min
        slope.append(df_P['diff'][top:i].mean())
    
    for i in range(0,int_min):
        slope.insert(i, 'NaN')

    return slope

def make_func_caller_find_lag_slope(df_P, min_time, max_time, interval, column):
    i = min_time
    while i <= max_time:    
        df_P[str(column) + '_slope' + '_lag_' + str(i)] = find_lag_slope(df_P, i, column)      
        i += interval
    return df_P 

def find_lead_slope(df_P, int_min, data_col):
    df_P['diff'] = df_P[data_col].diff()
    slope = []
    for i in range(0,len(df_P[data_col])-int_min):
        top = i + int_min
        slope.append(df_P['diff'][i:top].mean())
    
    for i in range(len(df_P[data_col])-int_min, len(df_P[data_col])):
        slope.insert(i, 'NaN')

    return slope

def make_func_caller_find_lead_slope(df_P, min_time, max_time, interval, column):
    i = min_time
    while i <= max_time:    
        df_P[str(column) + '_slope' + '_lead_' + str(i)] = find_lead_slope(df_P, i, column)      
        i += interval
    return df_P

def find_lag_integral(df_P, int_min, data_col):
    area_curve = []

    for i in range(int_min,len(df_P[data_col])):
        top = i - int_min
        area_curve.append(np.trapz(df_P[data_col][top:i]))
    
    for i in range(0,int_min):
        area_curve.insert(i, 'NaN')
    
    return area_curve


def find_lead_integral(df_P, int_min, data_col):
    area_curve = []

    for i in range(0,len(df_P[data_col])-int_min):
        top = i + int_min
        area_curve.append(np.trapz(df_P[data_col][i:top]))
    
    for i in range(len(df_P[data_col])-int_min, len(df_P[data_col])):
        area_curve.insert(i, 'NaN')
    
    return area_curve

#relative start time must be less than relative end time
def sliding_integral_lag(df_P, start_min_before, end_min_before, column):
    interval = start_min_before - end_min_before 
    a = df_P['e2v03'].shift(end_min_before).values
    v = np.zeros(interval)
    v[0:interval] = 1
    out = np.convolve(a, v, 'valid')
    out = np.concatenate((np.array([float('nan')] * (interval-1)), out))
    return out

#relative start time must be less than relative end time
def sliding_integral_lead(df_P, start_min_after, end_min_after, column):
    interval = end_min_after - start_min_after 
    a = df_P[column].shift(start_min_after).values
    a = a[~np.isnan(a)]
    v = np.zeros(end_min_after)
    v[0:interval] = 1
    out = np.convolve(a, v, 'valid')
    out = np.concatenate((out, np.array([float('nan')] * (start_min_after+end_min_after-1))))
    return out

def make_func_caller_find_lag_integral(df_P, min_time, max_time, interval, column):
    i = min_time
    while i <= max_time:    
        df_P[str(column) + '_int' + '_lag_' + str(i)] = find_lag_integral(df_P, i, column)      
        i += interval
    return df_P  

def make_func_caller_find_lead_integral(df_P, min_time, max_time, interval, column):
    i = min_time
    while i <= max_time:    
        df_P[str(column) + '_int' + '_lead_' + str(i)] = find_lead_integral(df_P, i, column)      
        i += interval
    return df_P 

def make_func_caller_sliding_integral_lag(df_P, max_end_min_before, min_end_min_before, frequency, interval_size, column):
    i = min_end_min_before
    while i <= max_end_min_before:
        start = i + interval_size
        df_P[str(column) + '_int' + '_slide_' + str(start) + '_to_' + str(i) + '_lag'] = sliding_integral_lag(df_P, start, i, column)      
        i += frequency
    return df_P 

def make_func_caller_sliding_integral_lead(df_P, min_start_min_after, max_start_min_after, frequency, interval_size, column):
    i = min_start_min_after
    while i <= max_start_min_after:
        start = i + interval_size
        df_P[str(column) + '_int' + '_slide_' + str(i) + '_to_' + str(start) + '_lead'] = sliding_integral_lead(df_P, start, i, column)      
        i += frequency
    return df_P 



 

def make_func_caller_find_sliding_slope(df_P, start_bef_or_aft_point, end_bef_or_aft_point, size, column):
    i = size
    while i <= size:    
        df_P[str(column) + '_slope' + '_slide_' + str(i)] = find_sliding_slope(df_P, start_bef_or_aft_point, end_bef_or_aft_point, i, column)      
        i += size
    return df_P