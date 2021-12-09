import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from file_paths import StdMotor
from preprocessing import load_scores,filt_data,load_data
from preprocessing import gait_data_dense,ftn_data_dense
import os
from feature_selection import FtnFeatures, GaitFeatures
from classifers import RFClassifier

dt=30
temp_saving_path=r'/Users/tianhaolei/Desktop'

def hist_labels(task,task_name):
    xlabel = [0, 1, 2, 3, 4]

    Overall = []
    Tremor_L = []
    Tremor_R = []
    Brady_L = []
    Brady_R = []
    Dys_L = []
    Dys_R = []

    Overall.append(sum(task['Overall'] == 0))
    Overall.append(sum(task['Overall'] == 1))
    Overall.append(sum(task['Overall'] == 2))
    Overall.append(sum(task['Overall'] == 3))
    Overall.append(sum(task['Overall'] == 4))
    Brady_L.append(sum(task['Bradykinesia - Left'] == 0))
    Brady_L.append(sum(task['Bradykinesia - Left'] == 1))
    Brady_L.append(sum(task['Bradykinesia - Left'] == 2))
    Brady_L.append(sum(task['Bradykinesia - Left'] == 3))
    Brady_L.append(sum(task['Bradykinesia - Left'] == 4))
    Brady_R.append(sum(task['Bradykinesia - Right'] == 0))
    Brady_R.append(sum(task['Bradykinesia - Right'] == 1))
    Brady_R.append(sum(task['Bradykinesia - Right'] == 2))
    Brady_R.append(sum(task['Bradykinesia - Right'] == 3))
    Brady_R.append(sum(task['Bradykinesia - Right'] == 4))
    Dys_L.append(sum(task['Dyskinesia - Left'] == 0))
    Dys_L.append(sum(task['Dyskinesia - Left'] == 1))
    Dys_L.append(sum(task['Dyskinesia - Left'] == 2))
    Dys_L.append(sum(task['Dyskinesia - Left'] == 3))
    Dys_L.append(sum(task['Dyskinesia - Left'] == 4))
    Dys_R.append(sum(task['Dyskinesia - Right'] == 0))
    Dys_R.append(sum(task['Dyskinesia - Right'] == 1))
    Dys_R.append(sum(task['Dyskinesia - Right'] == 2))
    Dys_R.append(sum(task['Dyskinesia - Right'] == 3))
    Dys_R.append(sum(task['Dyskinesia - Right'] == 4))
    Tremor_L.append(sum(task['Tremor - Left'] == 0))
    Tremor_L.append(sum(task['Tremor - Left'] == 1))
    Tremor_L.append(sum(task['Tremor - Left'] == 2))
    Tremor_L.append(sum(task['Tremor - Left'] == 3))
    Tremor_L.append(sum(task['Tremor - Left'] == 4))
    Tremor_R.append(sum(task['Tremor - Right'] == 0))
    Tremor_R.append(sum(task['Tremor - Right'] == 1))
    Tremor_R.append(sum(task['Tremor - Right'] == 2))
    Tremor_R.append(sum(task['Tremor - Right'] == 3))
    Tremor_R.append(sum(task['Tremor - Right'] == 4))

    plt.figure(figsize=(6, 12))
    plt.subplot(711)
    plt.bar(xlabel, Overall)
    plt.ylabel('frequency')
    plt.title(task_name)
    plt.legend(['Overall'])
    plt.subplot(712)
    plt.bar(xlabel, Tremor_L, color=['purple'])
    plt.ylabel('frequency')
    plt.legend(['Tremor - Left'])
    plt.subplot(713)
    plt.bar(xlabel, Tremor_R, color=['orange'])
    plt.ylabel('frequency')
    plt.legend(['Tremor - Right'])
    plt.subplot(714)
    plt.bar(xlabel, Brady_L, color=['g'])
    plt.ylabel('frequency')
    plt.legend(['Bradykinesia - Left'])
    plt.subplot(715)
    plt.bar(xlabel, Brady_R, color=['c'])
    plt.ylabel('frequency')
    plt.legend(['Bradykinesia - Right'])
    plt.subplot(716)
    plt.bar(xlabel, Dys_L, color=['m'])
    plt.ylabel('frequency')
    plt.legend(['Dyskinesia - Left'])
    plt.subplot(717)
    plt.bar(xlabel, Dys_R, color=['r'])
    plt.ylabel('frequency')
    plt.legend(['Dyskinesia - Right'])
    plt.xlabel('Standardized Motor Score')

    path=os.path.join(temp_saving_path,task_name+'_frequency.png')
    plt.savefig(path)
    return

def draw_signal_ftn(data_name,subject_id,activity,timepoint):
    raw_data = load_data(data_name)
    data = filt_data(raw_data)
    ff=FtnFeatures(data)
    patient=ff.data[(ff.data['subject_id']==subject_id)&
                    (ff.data['activity']==activity)&
                    (ff.data['timepoint']==timepoint)]
    index=int(patient.index.values[0])
    distance=ff.ftn_distance()
    elbow_angle=ff.elbow_angle()
    their_distance=distance[index]
    their_angle=elbow_angle[index]

    length=len(their_distance)
    x=np.linspace(0,length/dt,length)

    plt.figure(figsize=(12,6))
    plt.subplot(121)
    plt.plot(x,their_distance)
    plt.legend([f'{subject_id}_{activity}_{timepoint}'])
    plt.title('finger to nose distance')
    plt.xlabel('time (s)')
    plt.ylabel('distance [AU]')

    plt.subplot(122)
    plt.plot(x, their_angle)
    plt.legend([f'{subject_id}_{activity}_{timepoint}'])
    plt.title('elbow angle')
    plt.xlabel('time (s)')
    plt.ylabel('angle (radius)')

    path = os.path.join(temp_saving_path, f'{subject_id}_{timepoint}_ftn_distance.png')
    plt.savefig(path)

def hist_features_ftn(data_name,data_type:str,label='Overall'):
    raw_data = load_data(data_name)
    data = filt_data(raw_data)
    ff = FtnFeatures(data)
    _=ff.ftn_frequency()
    _=ff.avg_ang_speed()
    _=ff.med_ang_speed()
    _=ff.med_ang_accel()
    _=ff.cv_interval()

    feature_list=[
        'ftn_frequency',
        'avg_ang_speed',
        'med_ang_speed',
        'med_ang_accel',
        'cv_interval',
    ]
    rfc = RFClassifier(ff.data)
    # assigning the independent variables (features) for the classifier
    rfc.indie_var=feature_list

    # task can be 'Wlkg', 'Ftn', 'Ram'
    rfc.task = 'Ftn' # announce task first, then label second
    # labels can be 'Overall', 'Bradykinesia', 'Dyskinesia'
    rfc.labels=label

    indie_var=rfc.indie_var
    labels=rfc.labels
    indie_var['labels']=labels
    sns.pairplot(indie_var,hue='labels',dropna=True)

    # plt.figure(figsize=(6,8))
    # plt.subplots_adjust(wspace=0.2,hspace=0.3)
    # plt.subplot(321)
    # plt.hist(ff.data['ftn_frequency'])
    # plt.ylabel('frequency')
    # plt.title('ftn frequency')
    #
    # plt.subplot(322)
    # plt.hist(ff.data['avg_ang_speed'],color='green')
    # plt.title('avg_ang_speed')
    #
    # plt.subplot(323)
    # plt.hist(ff.data['med_ang_speed'],color='orange')
    # plt.title('med_ang_speed')
    # plt.ylabel('frequency')
    #
    # plt.subplot(324)
    # plt.hist(ff.data['med_ang_accel'],color='m')
    # plt.title('med_ang_accel')
    #
    # plt.subplot(325)
    # plt.hist(ff.data['cv_interval'],color='purple')
    # plt.title('cv_interval')
    # plt.ylabel('frequency')

    path=os.path.join(temp_saving_path,data_type+'_feature_hist.png')
    plt.savefig(path)

def hist_features_gait(data_name,data_type:str,label='Overall'):
    raw_data = load_data(data_name)
    data = filt_data(raw_data)
    gf = GaitFeatures(data)
    _=gf.step_frequency()
    _=gf.median_velocity()
    _=gf.median_amplitude()
    _=gf.accel_l_ankle()
    _=gf.accel_r_ankle()
    _=gf.cv_dist_ankle()

    feature_list=[
        'step_frequency',
        'median_velocity',
        'median_amplitude',
        'accel_r_ankle',
        'accel_l_ankle',
        'cv_dist_ankle'
    ]
    rfc = RFClassifier(gf.data)
    # assigning the independent variables (features) for the classifier
    rfc.indie_var=feature_list

    # task can be 'Wlkg', 'Ftn', 'Ram'
    rfc.task = 'Wlkg' # announce task first, then label second
    # labels can be 'Overall', 'Bradykinesia', 'Dyskinesia'
    rfc.labels=label

    indie_var=rfc.indie_var
    labels=rfc.labels
    indie_var['labels']=labels
    sns.pairplot(indie_var,hue='labels',dropna=True)


    # plt.figure(figsize=(6,8))
    # plt.subplots_adjust(wspace=0.2,hspace=0.3)
    # plt.subplot(321)
    # plt.hist(gf.data['step_frequency'])
    # plt.ylabel('frequency')
    # plt.title('step_frequency')
    #
    # plt.subplot(322)
    # plt.hist(gf.data['median_velocity'],color='green')
    # plt.title('median_velocity')
    #
    # plt.subplot(323)
    # plt.hist(gf.data['median_amplitude'],color='orange')
    # plt.title('median_amplitude')
    # plt.ylabel('frequency')
    #
    # plt.subplot(324)
    # plt.hist(gf.data['accel_l_ankle'],color='m')
    # plt.title('accel_l_ankle')
    #
    # plt.subplot(325)
    # plt.hist(gf.data['accel_r_ankle'],color='purple')
    # plt.title('accel_r_ankle')
    # plt.ylabel('frequency')
    #
    # plt.subplot(326)
    # plt.hist(gf.data['cv_dist_ankle'],color='indigo')
    # plt.title('cv_dist_ankle')

    path=os.path.join(temp_saving_path,data_type+'_feature_hist.png')
    plt.savefig(path)


def confusion_matrix_gait(data_name,label='Overall'):
    raw_data = load_data(data_name)
    data = filt_data(raw_data)
    gf = GaitFeatures(data)
    _ = gf.step_frequency()
    _ = gf.median_velocity()
    _ = gf.median_amplitude()
    _ = gf.accel_l_ankle()
    _ = gf.accel_r_ankle()
    _ = gf.cv_dist_ankle()

    feature_list = [
        'step_frequency',
        'median_velocity',
        'median_amplitude',
        'accel_r_ankle',
        'accel_l_ankle',
        'cv_dist_ankle'
    ]
    rfc = RFClassifier(gf.data)
    # assigning the independent variables (features) for the classifier
    rfc.indie_var = feature_list

    # task can be 'Wlkg', 'Ftn', 'Ram'
    rfc.task = 'Wlkg'  # announce task first, then label second
    # labels can be 'Overall', 'Bradykinesia', 'Dyskinesia'
    rfc.labels = label

    matrix=rfc(kfold=5,ordinal=True,confusion_m=True)
    index=['0: Normal','1: Slight','2: Mild','3: Moderate']
    matrix=pd.DataFrame(matrix,index=index,columns=index)
    plt.figure()
    sns.heatmap(matrix,annot=True)
    plt.xlabel('Model estimates')
    plt.ylabel('Clinical scores')
    plt.title('Gait')

    path = os.path.join(temp_saving_path, label + '_gait_confusion_matrix.png')
    plt.savefig(path)


def confusion_matrix_ftn(data_name,label='Overall'):
    raw_data = load_data(data_name)
    data = filt_data(raw_data)
    ff = FtnFeatures(data)
    _ = ff.ftn_frequency()
    _ = ff.avg_ang_speed()
    _ = ff.med_ang_speed()
    _ = ff.med_ang_accel()
    _ = ff.cv_interval()

    feature_list = [
        'ftn_frequency',
        'avg_ang_speed',
        'med_ang_speed',
        'med_ang_accel',
        'cv_interval',
    ]
    rfc = RFClassifier(ff.data)
    # assigning the independent variables (features) for the classifier
    rfc.indie_var = feature_list
    rfc = RFClassifier(ff.data)
    # assigning the independent variables (features) for the classifier
    rfc.indie_var = feature_list

    # task can be 'Wlkg', 'Ftn', 'Ram'
    rfc.task = 'Ftn'  # announce task first, then label second
    # labels can be 'Overall', 'Bradykinesia', 'Dyskinesia'
    rfc.labels = label

    matrix=rfc(kfold=5,ordinal=True,confusion_m=True)
    index=['0: Normal','1: Slight','2: Mild','3: Moderate']
    matrix=pd.DataFrame(matrix,index=index,columns=index)
    plt.figure()
    sns.heatmap(matrix,annot=True)
    plt.xlabel('Model estimates')
    plt.ylabel('Clinical scores')
    plt.title('Ftn')

    path = os.path.join(temp_saving_path, label + '_ftn_confusion_matrix.png')
    plt.savefig(path)

def feature_importance_ftn(data_name,label='Overall'):
    raw_data = load_data(data_name)
    data = filt_data(raw_data)
    ff = FtnFeatures(data)
    _ = ff.ftn_frequency()
    _ = ff.avg_ang_speed()
    _ = ff.med_ang_speed()
    _ = ff.med_ang_accel()
    _ = ff.cv_interval()

    feature_list = [
        'ftn_frequency',
        'avg_ang_speed',
        'med_ang_speed',
        'med_ang_accel',
        'cv_interval',
    ]
    rfc = RFClassifier(ff.data)
    # assigning the independent variables (features) for the classifier
    rfc.indie_var = feature_list
    rfc = RFClassifier(ff.data)
    # assigning the independent variables (features) for the classifier
    rfc.indie_var = feature_list

    # task can be 'Wlkg', 'Ftn', 'Ram'
    rfc.task = 'Ftn'  # announce task first, then label second
    # labels can be 'Overall', 'Bradykinesia', 'Dyskinesia'
    rfc.labels = label

    _=rfc(kfold=0, ordinal=True)
    feature_imp=np.mean(np.array(rfc.clf.feature_importances),axis=0)
    feature_imp = pd.Series(feature_imp, index=rfc.u_keys)

    print('\n', feature_imp)

    plt.figure()
    sns.barplot(x=feature_imp, y=feature_imp.index)
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title("Features importance")
    plt.subplots_adjust(left=0.2)
    path = os.path.join(temp_saving_path, label + '_ftn_feature_importance.png')
    plt.savefig(path)


def feature_importance_gait(data_name,label='Overall'):
    raw_data = load_data(data_name)
    data = filt_data(raw_data)
    gf = GaitFeatures(data)
    _ = gf.step_frequency()
    _ = gf.median_velocity()
    _ = gf.median_amplitude()
    _ = gf.accel_l_ankle()
    _ = gf.accel_r_ankle()
    _ = gf.cv_dist_ankle()

    feature_list = [
        'step_frequency',
        'median_velocity',
        'median_amplitude',
        'accel_r_ankle',
        'accel_l_ankle',
        'cv_dist_ankle'
    ]
    rfc = RFClassifier(gf.data)
    # assigning the independent variables (features) for the classifier
    rfc.indie_var = feature_list

    # task can be 'Wlkg', 'Ftn', 'Ram'
    rfc.task = 'Wlkg'  # announce task first, then label second
    # labels can be 'Overall', 'Bradykinesia', 'Dyskinesia'
    rfc.labels = label

    _=rfc(kfold=0, ordinal=True)
    feature_imp=np.mean(np.array(rfc.clf.feature_importances),axis=0)
    feature_imp = pd.Series(feature_imp, index=rfc.u_keys)

    print('\n', feature_imp)

    plt.figure()
    sns.barplot(x=feature_imp, y=feature_imp.index)
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title("Features importance")
    plt.subplots_adjust(left=0.2)
    path = os.path.join(temp_saving_path, label + '_gait_feature_importance.png')
    plt.savefig(path)



if __name__ == '__main__':
    labels=load_scores(StdMotor)
    xlabel=[0,1,2,3,4]

    Overall = []
    Tremor_L = []
    Tremor_R = []
    Brady_L = []
    Brady_R = []
    Dys_L = []
    Dys_R = []

    gait = labels.loc[labels['TaskAbb'] == 'Wlkg']
    FtnL = labels.loc[labels['TaskAbb'] == 'FtnL']
    FtnR = labels.loc[labels['TaskAbb'] == 'FtnR']

    # save the image of histogram of all the labels
    #hist_labels(FtnR,'Ftn - Right')

    # draw the signal of ftn
    #draw_signal_ftn(ftn_data_dense,subject_id=1043,activity='FtnR',timepoint=1)

    # draw histogram of all the features
    #hist_features_ftn(ftn_data_dense,'ftn_brady_R')
    #hist_features_gait(gait_data_dense, 'gait_brady_R')

    # confusion matrix of accuracy
    #confusion_matrix_ftn(ftn_data_dense,label='Bradykinesia')

    # feature importance
    #feature_importance_ftn(ftn_data_dense)
    feature_importance_gait(gait_data_dense)




