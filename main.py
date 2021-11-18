# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

from feature_selection import GaitFeatures,FtnFeatures
from preprocessing import load_data,filt_data
from preprocessing import gait_data_dense,ftn_data_dense,ram_data_dense
from preprocessing import ftn_data,ftn_data_stdscore
from classifers import RFClassifier
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    raw_data_gait=load_data(gait_data_dense)
    filter_data_gait=filt_data(raw_data_gait)

    raw_data_ftn=load_data(ftn_data_dense)
    filter_data_ftn=filt_data(raw_data_ftn)

    raw_data_ram = load_data(ram_data_dense)
    filter_data_ram=filt_data(raw_data_ram)

    gf=GaitFeatures(raw_data_gait)
    ftnf=FtnFeatures(raw_data_ftn)

    leg_ratio_difference=gf.leg_ratio_difference()
    vertical_angle_body = gf.vertical_angle_body()
    horizontal_angle_ankles=gf.horizontal_angle_ankles()
    horizontal_angle_wrists=gf.horizontal_angle_wrists()
    horizontal_dist_ankle=gf.horizontal_dist_ankle()
    speed_l_ankle=gf.speed_l_ankle()



    feature1=gf.step_frequency()
    feature2=gf.median_velocity()
    feature3=gf.median_amplitude()
    feature4=gf.accel_r_ankle()
    feature5=gf.accel_l_ankle()
    feature6=gf.cv_dist_ankle()

    '''
    feature_list=[
        'step_frequency',
        'median_velocity',
        'median_amplitude',
        'accel_r_ankle',
        'accel_l_ankle',
        'cv_dist_ankle'
    ]

    rfc=RFClassifier(gf.data)
    rfc.indie_var=feature_list
    rfc.labels='Overall'
    rfc.task='Wlkg'
    #rfc.med='1'
    rfc(kfold=5)
    '''

    measurement=vertical_angle_body

    for i in range(5):
        peak1,_ = find_peaks(measurement[i],distance=15,)
        peak2, _ = find_peaks(-measurement[i], distance=15, )

        plt.figure()
        plt.plot(measurement[i])
        plt.plot(peak1, measurement[i][peak1], 'x')
        plt.plot(peak2, measurement[i][peak2], 'o')
    '''

    plt.figure()
    plt.hist(feature1)
    plt.title('step frequency')
    plt.figure()
    plt.hist(feature2)
    plt.title('median velocity')
    plt.figure()
    plt.hist(feature3)
    plt.title('median amplitude')
    plt.figure()
    plt.hist(feature4)
    plt.title('accel_l_ankle')
    plt.figure()
    plt.hist(feature5)
    plt.title('accel_r_ankle')
    plt.figure()
    plt.hist(feature6)
    plt.title('cv_dist_ankle')
    '''

    plt.show()





