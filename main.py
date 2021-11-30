
from feature_selection import GaitFeatures,FtnFeatures,RamFeatures
from preprocessing import load_data,filt_data
from preprocessing import gait_data_dense,ftn_data_dense,ram_data_dense
from preprocessing import ftn_data,ftn_data_stdscore
from classifers import RFClassifier
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import numpy as np
import os

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # filtered data for gait analysis
    raw_data_gait=load_data(gait_data_dense)
    filter_data_gait=filt_data(raw_data_gait)

    # filtered data for ftn analysis
    raw_data_ftn=load_data(ftn_data_dense)
    filter_data_ftn=filt_data(raw_data_ftn)

    # filtered data for ram analysis
    raw_data_ram = load_data(ram_data_dense)
    filter_data_ram=filt_data(raw_data_ram)

    # instantiate a feature object
    gf=GaitFeatures(raw_data_gait)
    ftn=FtnFeatures(raw_data_ftn)
    ram=RamFeatures(raw_data_ram)

    ##########################
    # signal for gait features
    ##########################
    # leg_ratio_difference=gf.leg_ratio_difference()
    # vertical_angle_body = gf.vertical_angle_body()
    # horizontal_angle_ankles=gf.horizontal_angle_ankles()
    # horizontal_angle_wrists=gf.horizontal_angle_wrists()
    # horizontal_dist_ankle=gf.horizontal_dist_ankle()
    # speed_l_ankle=gf.speed_l_ankle()

    ###############
    # gait features
    ###############
    # feature1=gf.step_frequency()
    # feature2=gf.median_velocity()
    # feature3=gf.median_amplitude()
    # feature4=gf.accel_r_ankle()
    # feature5=gf.accel_l_ankle()
    # feature6=gf.cv_dist_ankle()

    #########################
    # signals for ftn features
    #########################
    elbow_angle = ftn.elbow_angle()
    p_t_elbow_angle=ftn.p_t_elbow_angle()

    ###############
    # ftn features
    ###############
    ftn_frequency=ftn.ftn_frequency()
    avg_ang_speed=ftn.avg_ang_speed()
    med_ang_speed=ftn.med_ang_speed()
    med_ang_accel= ftn.med_ang_accel()
    cv_interval=ftn.cv_interval()

    # feature list for gait analysis
    # feature_list=[
    #     'step_frequency',
    #     'median_velocity',
    #     'median_amplitude',
    #     'accel_r_ankle',
    #     'accel_l_ankle',
    #     'cv_dist_ankle'
    # ]

    # feature list for ftn analysis
    feature_list = [
        'ftn_frequency',
        'avg_ang_speed',
        'med_ang_speed',
        'med_ang_accel',
        'cv_interval',
    ]

    # feed the data to the classifier to the classification
    rfc = RFClassifier(ftn.data)
    # or change the dataset by assigning it to self.data
    rfc.data=ftn.data

    # assigning the independent variables (features) for the classifier
    rfc.indie_var=feature_list

    # task can be 'Wlkg', 'Ftn', 'Ram'
    rfc.task = 'Ftn' # announce task first, then label second
    # labels can be 'Overall', 'Bradykinesia', 'Dyskinesia'
    rfc.labels='Overall'
    rfc.med='1'
    rfc(kfold=5,ordinal=True)

    ##################
    # plotting signals
    ##################
    # measurement={some signals}
    # measurement = elbow_angle
    # for i in range(50):
    #     peak1,_ = find_peaks(measurement[i],height=np.nanmean(measurement[i]), distance=25,)
    #     peak2, _ = find_peaks(-measurement[i],height=np.nanmean(-measurement[i]), distance=25, )
    #     labels = rfc.labels
    #     subj=rfc.data['subject_id']
    #     task=rfc.data['activity']
    #
    #     plt.figure()
    #     plt.plot(measurement[i])
    #     plt.plot(peak1, measurement[i][peak1], 'x')
    #     plt.plot(peak2, measurement[i][peak2], 'o')
    #     plt.xlim([0, 600])
    #     plt.title(str(labels[i])+str(subj[i])+str(task[i]))
    # '''
    #
    # plt.figure()
    # plt.hist(feature1)
    # plt.title('step frequency')
    # plt.figure()
    # plt.hist(feature2)
    # plt.title('median velocity')
    # plt.figure()
    # plt.hist(feature3)
    # plt.title('median amplitude')
    # plt.figure()
    # plt.hist(feature4)
    # plt.title('accel_l_ankle')
    # plt.figure()
    # plt.hist(feature5)
    # plt.title('accel_r_ankle')
    # plt.figure()
    # plt.hist(feature6)
    # plt.title('cv_dist_ankle')
    # '''
    #
    # plt.show()





