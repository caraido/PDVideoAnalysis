import numpy as np
import pandas as pd
import os
from file_paths import pose_data_path, raw_scores_path
from file_paths import gait_data_dense
from file_paths import ftn_data,ftn_data_dense,ftn_data_stdscore
from file_paths import ram_data,ram_data_dense,ram_data_stdscore
from file_paths import StdMotor


def load_data(data_type):
    '''
    :param data_type: data/data_dense/data_stdscore
    :return: a dictionary of data (refer to file_paths.py)
    '''
    data_path = os.path.join(pose_data_path, data_type)
    data=np.load(data_path,allow_pickle=True)
    return data


def load_scores(score_type):
    '''
    :param score_type: here is StdMotor
    :return: a DataFrame structure
    '''
    scores_path=os.path.join(raw_scores_path,score_type)
    scores=pd.read_csv(scores_path)
    return scores

def filt_scores(score:pd.DataFrame):
    for i in range(len(score)):
        value=score['Visit'].iloc[[i]]
        if 'Month' in str(value):
            score['Visit'].iloc[[i]]=7
        elif '150' in str(value):
            score['Visit'].iloc[[i]] = 6
        elif '120' in str(value):
            score['Visit'].iloc[[i]] = 5
        elif '90' in str(value):
            score['Visit'].iloc[[i]] = 4
        elif '60' in str(value):
            score['Visit'].iloc[[i]] = 3
        elif '30' in str(value):
            score['Visit'].iloc[[i]] = 2
        else:
            score['Visit'].iloc[[i]] = 1
    return score

def filt_data(raw_data:pd.DataFrame):
    '''
    :return: filter out the baylor data which subjIDs have 3 digits from James' data
    '''
    data=pd.DataFrame(raw_data['data'])
    subjID=data['subject_id']
    boolean_filtered_subjID = [bool(i//1000) for i in subjID]
    filtered_subjiD=subjID.loc[boolean_filtered_subjID]
    filtered_data=data[data['subject_id'].isin(filtered_subjiD)]
    raw_data['data']=filtered_data
    return raw_data




