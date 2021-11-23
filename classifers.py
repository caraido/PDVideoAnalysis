import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,KFold
from sklearn import metrics
import seaborn as sns
from preprocessing import load_scores,StdMotor,filt_scores
from utils import emit_nan
import numpy as np


class OrdinalClassifier:

    def __init__(self,clf):
        self.clf=clf
        self.clfs={}

    def fit(self,X,y):
        self.unique_class=np.sort(np.unique(y))
        if self.unique_class.shape[0]>2:
            for i in range(self.unique_class.shape[0]-1):
                # for each k-1 ordinal value we fit a binary classfication problem
                binary_y=(y>self.unique_class[i]).astype(np.uint8)
                clf=clone(self.clf)
                clf.fit(X,binary_y)
                self.clfs[i]=clf

    def predict_proba(self,X):
        clfs_predict={k: self.clfs[k].predict_proba(X) for k in self.clfs}
        predicted=[]
        for i, y in enumerate(self.unique_class):
            if i==0:
                # V1=1-Pr(y>V1)
                predicted.append(1-clfs_predict[y][:,1])
            elif y in clfs_predict:
                # Vi = Pr(y>Vi-1) - Pr(y>Vi)
                predicted.append(clfs_predict[y-1][:,1]-clfs_predict[y][:,1])
            else:
                # Vk=Pr(y>Vk-1)
                predicted.append(clfs_predict[y-1][:,1])
        return np.vstack(predicted).T

    def predict(self,X):
        return np.argmax(self.predict_proba(X),axis=1)


class RFClassifier:
    # random forest classifier
    def __init__(self,data:pd.DataFrame):
        self.data=data
        self.keys=self.data.keys()
        self.u_keys=None
        self.n_Data=len(data)
        self.med = None # can be on of 1,2,3,4,5,6,7
        self.task='Wlkg' # also could be "FtnL" "FtnR" "RamL" RamR"

        self._indie_var=None
        self._labels=None

    @property
    def indie_var(self):
        return self._indie_var

    @indie_var.setter
    def indie_var(self,u_keys:list):
        self.u_keys=u_keys
        all_keys=set(self.keys.tolist())
        if not set(u_keys).issubset(all_keys):
            raise Exception('key(s) not found!')
        else:
            self._indie_var=self.data[u_keys]

    @property
    def labels(self):

        return self._labels

    @labels.setter
    def labels(self,label_name:str):
        if label_name not in self.keys:
            raw_scores= load_scores(StdMotor)
            std_scores=filt_scores(raw_scores)
            id=self.data['subject_id']
            if self.task=='Ftn':
                tasks=['FtnL','FtnR']
                filtered = std_scores[std_scores['TaskAbb'].isin(tasks)][std_scores['SubjID'].isin(id)]
            else:
                filtered=std_scores[std_scores['TaskAbb']==self.task][std_scores['SubjID'].isin(id)]

            scores=[]
            for i in range(len(self.data)):
                SubjID=id.iloc[[i]]
                visit=self.data['timepoint'].iloc[[i]]
                side=self.data['activity'].iloc[[i]]
                result = filtered.loc[(filtered['SubjID'].values==SubjID.values)&
                                      (filtered['Visit'].values==visit.values)&
                                      (filtered['TaskAbb'].values==side.values)]
                if len(result)==1:
                    scores.append(result[label_name].tolist()[0])
                else:
                    raise Exception("cannot match the event in the scores! check the 'SubjId,' 'Visit' and 'TaskAbb")

            self.data[label_name]=scores
        self._labels=self.data[label_name]

    def __call__(self,kfold=0,ordinal=True):
        indie_var,labels=emit_nan(self.indie_var,self.labels)

        if not kfold:
            X_train, X_test,y_train,y_test=train_test_split(indie_var,labels,test_size=0.2)

            if ordinal:
                clf=OrdinalClassifier(RandomForestClassifier(n_estimators=100))
            else:
                clf=RandomForestClassifier(n_estimators=100)
            clf.fit(X_train,y_train)

            y_pred=clf.predict(X_test)

            print("Accuracy: ",metrics.accuracy_score(y_test,y_pred))


        else:
            kf=KFold(n_splits=kfold,shuffle=True)
            if ordinal:
                clf = OrdinalClassifier(RandomForestClassifier(n_estimators=100))
            else:
                clf = RandomForestClassifier(n_estimators=100)

            acc_score=[]

            for train_index,test_index in kf.split(indie_var):
                X_train,X_test=indie_var.iloc[train_index,:],indie_var.iloc[test_index,:]
                y_train,y_test=np.array(labels)[train_index],np.array(labels)[test_index]

                clf.fit(X_train,y_train)
                y_pred=clf.predict(X_test)

                acc=metrics.accuracy_score(y_test, y_pred)
                acc_score.append(acc)

            avg_acc_score=sum(acc_score)/kfold

            print('accuracy of each fold - {}'.format(acc_score))
            print('Avg accuracy: {}'.format(avg_acc_score))


        '''
        feature_imp = pd.Series(clf.feature_importances_, index=self.u_keys)

        print('\n', feature_imp)

        sns.barplot(x=feature_imp, y=feature_imp.index)

        plt.xlabel('Feature Importance Score')
        plt.ylabel('Features')
        plt.title("Visualizing Important Features")
        plt.show()
        '''


if __name__=='__main__':
    pass
