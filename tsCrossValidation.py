# Time Series Cross Validation
import numpy as np
import pandas as pd
from itertools import product
import time

'''
Description: Multiple Splits Cross Validation on Time Series data
Args:
    num: Number of DataSet
    n_splits: Split times
Return: 
    split_position_df: All set of splits position in a Pandas dataframe
'''
def mulTsCrossValidation(num, n_splits):
    split_position_lst = []
    # Calculate the split position for each time 
    for i in range(1, n_splits+1):
        # Calculate train size and test size
        train_size = i * num // (n_splits + 1) + num % (n_splits + 1)
        test_size = num //(n_splits + 1)

        # Calculate the start/split/end point for each fold
        start = 0
        split = train_size
        end = train_size + test_size
        
        # Avoid to beyond the whole number of dataSet
        if end > num:
            end = num
        split_position_lst.append((start,split,end))
        
    # Transform the split position list to a Pandas Dataframe
    split_position_df = pd.DataFrame(split_position_lst,columns=['start','split','end'])
    return split_position_df


'''
Description: Blocked Time Series Cross Validation
Args:
    num: Number of DataSet
    n_splits: Split times
Return: 
    split_position_df: All set of splits position in a Pandas dataframe
'''
def blockedTsCrossValidation(num, n_splits):
    kfold_size = num // n_splits

    split_position_lst = []
    # Calculate the split position for each time 
    for i in range(n_splits):
        # Calculate the start/split/end point for each fold
        start = i * kfold_size
        end = start + kfold_size
        # Manually set train-test split proportion in each fold
        split = int(0.8 * (end - start)) + start
        split_position_lst.append((start,split,end))
        
    # Transform the split position list to a Pandas Dataframe
    split_position_df = pd.DataFrame(split_position_lst,columns=['start','split','end'])
    return split_position_df


'''
Description: Walk Forward Validation on Time Series data
Args:
    num: Number of DataSet
    min_obser: Minimum Number of Observations
    expand_window: Sliding or Expanding Window
Return: 
    split_position_df: All set of splits position in a Pandas dataframe
'''
def wfTsCrossValidation(num, min_obser, expand_window):
    split_position_lst = []
    # Calculate the split position for each time 
    for i in range(min_obser,num,expand_window):
        # Calculate the start/split/end point for each fold
        start = 0
        split = i
        end = split + expand_window
        
        # Avoid to beyond the whole number of dataSet
        if end > num:
            end = num
        split_position_lst.append((start,split,end))
        
    # Transform the split position list to a Pandas Dataframe
    split_position_df = pd.DataFrame(split_position_lst,columns=['start','split','end'])
    return split_position_df

'''
Description: Apply calculations on Time Series Cross Validation results to form the final Model Comparison Table
Args:
    cv_result: The results from tsCrossValidation()
    model_info: The model information which you would like to show
    evaluator_lst: The evaluator metrics which you would like to show
Return: 
    comparison_df: A pandas dataframe of a model on a type of Time Series Cross Validation
'''
def modelComparison(cv_result, model_info, evaluator_lst):
    # Calculate mean of all splits on chosen evaluator 
    col_mean_df = cv_result[evaluator_lst].mean().to_frame().T
    # Extract model info
    model_info_df = cv_result[model_info][:1]
    # Concatenate by row
    comparison_df = pd.concat([model_info_df,col_mean_df],axis=1)
    return comparison_df