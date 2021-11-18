# it should have at least (1)gait feature (2)ftn feature (3) ram feature
# currently gait feature calculation is referred from Rupprechter et al 2021

import pandas as pd
import numpy as np
import vg
from scipy.signal import find_peaks
from scipy.stats import variation


class GaitFeatures:

    def __init__(self, data_dense):
        self.raw_data = data_dense
        self.data = pd.DataFrame(self.raw_data['data'])

        self.keypoints_list = self.data['keypoints'].to_list()
        self.keypoint_names = self.raw_data['keypoint_names']
        self.get_keypoint_indices()
        self.alpha=2
        self.beta=1

    def get_keypoint_indices(self):
        self.nose = self.keypoint_names.index('Nose')
        self.wrist_R = self.keypoint_names.index('Right Wrist')
        self.wrist_L = self.keypoint_names.index('Left Wrist')
        self.hip_R = self.keypoint_names.index('Right Hip')
        self.hip_L = self.keypoint_names.index('Left Hip')
        self.ankle_R = self.keypoint_names.index('Right Ankle')
        self.ankle_L = self.keypoint_names.index('Left Ankle')
        # self.heel_R=list(self.keypoint_names.keys()).index('Right Heel')
        # self.heel_L=list(self.keypoint_names.keys()).index('Left Heel')

    def leg_ratio_difference(self):
        # left leg/right leg-right leg/left leg

        left_hip = [i[:, self.hip_L, :] for i in self.keypoints_list]
        right_hip = [i[:, self.hip_R, :] for i in self.keypoints_list]
        left_ankle = [i[:, self.ankle_L, :] for i in self.keypoints_list]
        right_ankle = [i[:, self.ankle_R, :] for i in self.keypoints_list]

        dist_L = [np.linalg.norm(left_ankle[i] - left_hip[i],axis=1) for i in range(len(self.keypoints_list))]
        dist_R = [np.linalg.norm(right_ankle[i] - right_hip[i],axis=1) for i in range(len(self.keypoints_list))]
        this_value = [dist_L[i] / dist_R[i] - dist_R[i] / dist_L[i] for i in range(len(self.keypoints_list))]
        self.data['leg_ratio_difference'] = this_value
        return this_value

    def vertical_angle_body(self):
        # different from the reference since we're using 3D
        # here assume y is the vertical axis
        # TODO: check if y or z is the vertical axis

        this_value=[]
        nose=[i[:, self.nose, :] for i in self.keypoints_list]
        left_ankle = [i[:, self.ankle_L, :] for i in self.keypoints_list]
        right_ankle = [i[:, self.ankle_R, :] for i in self.keypoints_list]
        for i in range(len(self.keypoints_list)):
            nose_to_ankle=nose[i]-(left_ankle[i]+right_ankle[i])/2
            unit_nose_to_ankle=nose_to_ankle/np.linalg.norm(nose_to_ankle,axis=1)[:,np.newaxis]
            angle=np.arccos(np.dot(unit_nose_to_ankle,np.array([0,-1,0])))
            angle=angle[~np.isnan(angle)]
            this_value.append(angle)

        self.data['vertical_angle_body']=this_value
        return this_value

    def horizontal_angle_ankles(self):
        # this is defined differently than the one in the paper
        # instead of using the x axis as a reference, we would like to use the vector of hips to describe the direction the patient is facing
        # it should be arccos(dot(unit(hips), unit(ankles)))

        this_value = []
        left_hip = [i[:, self.hip_L, :] for i in self.keypoints_list]
        right_hip = [i[:, self.hip_R, :] for i in self.keypoints_list]
        left_ankle = [i[:, self.ankle_L, :] for i in self.keypoints_list]
        right_ankle = [i[:, self.ankle_R, :] for i in self.keypoints_list]
        for i in range(len(self.keypoints_list)):
            v_hip=left_hip[i]-right_hip[i]
            v_ankle=left_ankle[i]-right_ankle[i]

            unit_hip=v_hip/np.linalg.norm(v_hip,axis=1)[:,np.newaxis]
            unit_ankle =v_ankle / np.linalg.norm(v_ankle, axis=1)[:, np.newaxis]

            angle=vg.angle(unit_ankle,unit_hip, units='rad')
            #angle=np.arccos(np.dot(unit_ankle, np.array([1, 0, 0])))
            this_value.append(angle)

        self.data['horizontal_angle_ankles'] = this_value
        return this_value

    def horizontal_angle_wrists(self):
        # this is defined differently than the one in the paper

        this_value = []

        left_wrist = [i[:, self.wrist_L, :] for i in self.keypoints_list]
        right_wrist = [i[:, self.wrist_R, :] for i in self.keypoints_list]
        for i in range(len(self.keypoints_list)):
            v_wrist = left_wrist[i] - right_wrist[i]
            unit_wrist = v_wrist / np.linalg.norm(v_wrist, axis=1)[:, np.newaxis]

            angle = np.arccos(np.dot(unit_wrist, np.array([0, 1, 0])))
            angle = angle[~np.isnan(angle)]
            this_value.append(angle)

        self.data['horizontal_angle_wrists'] = this_value
        return this_value

    def horizontal_dist_ankle(self):
        # this should be distance between heels
        # we only have ankle data
        # normalized by the average leg length of the patient by calculate the mean value of y_center hip to y_center ankle distance across time

        left_hip = [i[:, self.hip_L, :] for i in self.keypoints_list]
        right_hip = [i[:, self.hip_R, :] for i in self.keypoints_list]
        left_ankle = [i[:, self.ankle_L, :] for i in self.keypoints_list]
        right_ankle = [i[:, self.ankle_R, :] for i in self.keypoints_list]

        this_value=[]
        for i in range(len(self.keypoints_list)):
            center_hip=(left_hip[i]+right_hip[i])/2
            center_ankle=(left_ankle[i]+right_ankle[i])/2
            center_hip_to_ground=center_hip[:,1] # assume y axis is direction of standing
            center_ankle_to_ground=center_ankle[:,1]
            avg_leg_len=np.average(np.abs(center_hip_to_ground-center_ankle_to_ground))
            distance=np.linalg.norm(left_ankle[i]-right_ankle[i],axis=1)/np.array(avg_leg_len)
            this_value.append(distance)

        self.data['horizontal_dist_ankle']=this_value
        return this_value

    def speed_l_ankle(self,frame_rate=15):
        if not isinstance(frame_rate,str):
            frame_rate=frame_rate*np.ones(len(self.keypoints_list))
        left_ankle = [i[:, self.ankle_L, :] for i in self.keypoints_list]
        speed=[np.linalg.norm(np.diff(left_ankle[i],axis=0),axis=1)*frame_rate[i] for i in range(len(self.keypoints_list))]
        self.data['speed_l_ankle'] =speed
        return speed

    def speed_r_ankle(self,frame_rate=15):
        if not isinstance(frame_rate,str):
            frame_rate=frame_rate*np.ones(len(self.keypoints_list))
        right_ankle = [i[:, self.ankle_R, :] for i in self.keypoints_list]
        speed=[np.linalg.norm(np.diff(right_ankle[i],axis=0),axis=1)*frame_rate[i] for i in range(len(self.keypoints_list))]
        self.data['speed_r_ankle'] =speed
        return speed

    def p_t_legs(self,frame_rate=15):
        if 'leg_ratio_difference' in self.data.keys():
            feature =self.data['leg_ratio_difference'].to_numpy()
        else:
            feature = self.leg_ratio_difference()
        peak = [find_peaks(feature[i],height=0,distance=frame_rate,)[0] for i in range(len(feature))]
        trough = [find_peaks(-feature[i],height=0,distance=frame_rate,)[0] for i in range(len(feature))]
        self.data['peak_legs'] = peak
        self.data['trough_legs'] = trough
        return peak,trough

    def p_t_ankle(self):
        pass

    def p_t_wrists(self,frame_rate=15):
        if 'horizontal_angle_wrists' in self.data.keys():
            feature =self.data['horizontal_angle_wrists'].to_numpy()
        else:
            feature = self.horizontal_angle_wrists()
        peak = [find_peaks(feature[i],distance=frame_rate,)[0] for i in range(len(feature))]
        trough = [find_peaks(-feature[i],distance=frame_rate,)[0] for i in range(len(feature))]
        self.data['peak_wrists'] = peak
        self.data['trough_wrists'] = trough
        return peak,trough

    def p_t_body(self):
        pass

    # feature 1
    def step_frequency(self,frame_rate=15):
        # only calculate
        if 'peak_legs' in self.data.keys() and 'trough_legs' in self.data.keys():
            peak=self.data['peak_legs']
            trough=self.data['trough_legs']
        else:
            peak,trough=self.p_t_legs()

        this_value=[]
        for i in range(len(peak)):
            n_peak=len(peak[i])
            n_trough=len(trough[i])
            alpha = self.alpha+n_peak+n_trough
            beta= self.beta+len(self.keypoints_list[i])/frame_rate
            step_frequency=alpha/beta
            this_value.append(step_frequency)

        self.data['step_frequency'] = this_value
        return this_value

    # feature 2
    def median_velocity(self,frame_rate=15):
        # median velocity of arm swing
        if 'horizontal_angle_wrists' in self.data.keys() :
            horizontal_angle_wrists=self.data['horizontal_angle_wrists'].to_list()
        else:
            horizontal_angle_wrists=self.horizontal_angle_wrists()
        median_velocity=[np.median(np.diff(horizontal_angle_wrists[i])/frame_rate) for i in range(len(self.keypoints_list))]

        self.data['median_velocity'] = median_velocity
        return median_velocity

    # feature 3
    def median_amplitude(self):
        # median amplitude of arm swing
        # due to the characteristic of the data, here the max(peak) and min(trough) data points are eliminated
        # this calculation is not accurate

        if 'peak_wrists' in self.data.keys() and 'trough_wrists' in self.data.keys():
            peak=self.data['peak_wrists']
            trough=self.data['trough_wrists']
        else:
            peak,trough=self.p_t_wrists()

        horizontal_angle_wrists=self.data['horizontal_angle_wrists']

        this_value=[]
        for i in range(len(self.keypoints_list)):
            this_peak=peak[i]
            this_trough=trough[i]

            peak_value=horizontal_angle_wrists[i][this_peak]
            trough_value = horizontal_angle_wrists[i][this_trough]
            peak_value.sort()
            temp_peak_value=peak_value[:-1]# emit the last

            trough_value.sort()
            temp_trough_value=trough_value[1:] # emit the first

            amplitude=(sum(temp_peak_value)+sum(temp_trough_value))/(len(this_peak)+len(this_trough))
            this_value.append(amplitude)
        self.data['median_amplitude']=this_value
        return this_value

    # feature 4
    def accel_l_ankle(self,):
        # median value of acceleration of left ankle
        if 'speed_l_ankle' in self.data.keys():
            feature =self.data['speed_l_ankle'].to_numpy()
        else:
            feature = self.horizontal_angle_wrists()

        this_value=[]
        for i in range(len(feature)):
            accel=np.diff(feature[i])/feature[i][:-1]
            accel_med=np.median(accel)
            this_value.append(accel_med)
        self.data['accel_l_ankle']=this_value
        return this_value

    # feature 5
    def accel_r_ankle(self, ):
        # median value of acceleration of right ankle
        if 'speed_r_ankle' in self.data.keys():
            feature = self.data['speed_r_ankle'].to_numpy()
        else:
            feature = self.horizontal_angle_wrists()

        this_value = []
        for i in range(len(feature)):
            accel = np.diff(feature[i]) / feature[i][:-1]
            accel_med = np.median(accel)
            this_value.append(accel_med)
        self.data['accel_r_ankle'] = this_value
        return this_value

    # feature 6
    def cv_dist_ankle(self):
        # coefficient of variation of distance of ankles
        if 'horizontal_dist_ankle' in self.data.keys():
            feature = self.data['horizontal_dist_ankle'].to_numpy()
        else:
            feature = self.horizontal_dist_ankle()

        this_value=[variation(feature[i]) for i in range(len(feature))]
        self.data['cv_dist_ankle']=this_value
        return this_value


class FtnFeatures:

    def __init__(self, data_dense):
        self.raw_data = data_dense
        self.data = pd.DataFrame(self.raw_data['data'])

        self.keypoints_list = self.data['keypoints'].to_list()
        self.keypoint_names = self.raw_data['keypoint_names']
        self.get_keypoint_indices()

    def get_keypoint_indices(self):
        self.nose = self.keypoint_names.index('Nose')
        self.wrist_R = self.keypoint_names.index('Right Wrist')
        self.wrist_L = self.keypoint_names.index('Left Wrist')
        self.hip_R = self.keypoint_names.index('Right Hip')
        self.hip_L = self.keypoint_names.index('Left Hip')
        self.ankle_R = self.keypoint_names.index('Right Ankle')
        self.ankle_L = self.keypoint_names.index('Left Ankle')
        # self.heel_R=list(self.keypoint_names.keys()).index('Right Heel')
        # self.heel_L=list(self.keypoint_names.keys()).index('Left Heel')

    # feature 1
    def ftn_frequency(self):
        # a measurement of tapping nose frequency
        pass

    # feature 2
    def med_ang_speed(self):
        # measure the average angular speed of each finger to nose trial. angle is formed by the arm and the forearm
        pass

    # feature 3
    def med_ang_accel(self):
        # measure the average acceleration of each finger to nose trial
        pass

    # feature 4
    def pause_time(self,threshold):
        # measure how many times the patient pauses the task
        pass


class RamFeature:

    # TODO: need to be changed
    def __init__(self, data_dense):
        self.raw_data = data_dense
        self.data = pd.DataFrame(self.raw_data['data'])

        self.keypoints_list = self.data['keypoints'].to_list()
        self.keypoint_names = self.raw_data['keypoint_names']
        self.get_keypoint_indices()

    # TODO: need to be changed
    def get_keypoint_indices(self):
        self.nose = self.keypoint_names.index('Nose')
        self.wrist_R = self.keypoint_names.index('Right Wrist')
        self.wrist_L = self.keypoint_names.index('Left Wrist')
        self.hip_R = self.keypoint_names.index('Right Hip')
        self.hip_L = self.keypoint_names.index('Left Hip')
        self.ankle_R = self.keypoint_names.index('Right Ankle')
        self.ankle_L = self.keypoint_names.index('Left Ankle')
        # self.heel_R=list(self.keypoint_names.keys()).index('Right Heel')
        # self.heel_L=list(self.keypoint_names.keys()).index('Left Heel')

    # calculate the flip of palm
    def flip(self):
        pass

    # feature 1
    def flip_ampl(self):
        # measure the amplitude of hand flipping
        pass

    # feature 2
    def flip_freq(self):
        # measure the frequency of flipping
        pass

    # feature 3
    def avg_flip_speed(self):
        pass

    # feature 4
    # the decrease for flipping amplitude

    # feature 5
    # the decrease for flipping frequency






