#!python3
#!/usr/bin/env python3


import paho.mqtt.client as mqtt
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import warnings, json
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier

from scipy.stats import mode

from flask import Flask
from flask import request

app = Flask(__name__)


classes_meaning= {1:'still_hands_back',2:'still_hands_side'}
    

def read_data(fname='all'):
    print('::::Read Data::::')
    freq = '19230U'   # ~19ms, using sampling frequency is 52samples/sec
    dt = datetime.now()
    data = pd.DataFrame()
    
    if fname == 'all':    # read all files
        data_files = []
        
        for f_id in range(1,3):
            data = pd.read_csv('data/%i.csv' % f_id, 
                            #names=['index','xL', 'yL', 'zL', 'xR', 'yR', 'zR', 'label'],
                            names=['index','xL', 'yL', 'zL', 'label'],   
                            header=None, index_col=0)
            data.index = pd.date_range(start=dt, periods=data.shape[0], freq=freq)
            dt += timedelta(hours=3)
            data_files.append(data)
            
            data = pd.concat(data_files)
    
    else:    # read one file only
        data = pd.read_csv('data/%s.csv' %fname, 
                   #names=['index','xL', 'yL', 'zL', 'xR', 'yR', 'zR', 'label'], 
                   names=['index','xL', 'yL', 'zL', 'label'],          
                   header=None, index_col=0)

        data.index = pd.date_range(start='00:00:00', periods=data.shape[0], freq=freq)
    
    # Filter and clean data
    data = data.dropna()
    data = data[data['label'] != 0]   # some rows are misclassified as 0
    
        
    return data

def rms(ts): return np.sqrt(np.mean(ts**2))

def corrL(df): 
    cor = df.corr()
    return pd.DataFrame({'xyL':[cor['xL']['yL']], 'xzL':[cor['xL']['zL']], 'yzL':[cor['yL']['zL']]})

def corrR(df): 
    cor = df.corr()
    return pd.DataFrame({'xyR':[cor['xR']['yR']], 'xzR':[cor['xR']['zR']], 'yzR':[cor['yR']['zR']]})

def get_simple_features(data, wsize='10s', f_list=['mean', 'std', 'var', rms]):
    print('::::START:::: Get Simple Features::::')
    # f_list is a list of features names or methods to apply in resampling
    
    # features that invlove one dimension only.
    fname = '_' + (f_list[0] if isinstance(f_list[0], str) else f_list[0].__name__)   
    feats = data[['xL','yL','zL']].resample(wsize, how=f_list[0]).add_suffix('%s_l' % fname)
    #feats = data[['xR','yR','zR']].resample(wsize, how=f_list[0]).add_suffix('%s_r' % fname)
    
    
    for i, f in enumerate(f_list[1:]):
        fname = '_' + (f if isinstance(f, str) else f.__name__)
        feat = data[['xL','yL','zL']].resample(wsize, how=f).add_suffix('%s_L' % fname)
        feats = feats.join(feat)
        #feat = data[['xR','yR','zR']].resample(wsize, how=f).add_suffix('%s_R' % fname)
        #feats = feats.join(feat) 

    # features that involve more than one dimension.                                               
    mean_mag = (data**2).sum(axis=1).resample(wsize, how=lambda ts: np.sqrt(ts).mean())
    mean_mag.name = 'mean_mag'
    feats = feats.join(mean_mag) 

    pairs_cor = data.groupby(pd.TimeGrouper(wsize)).apply(corrL).reset_index(1, drop=True)
    feats = feats.join(pairs_cor) 
    #pairs_cor = data.groupby(pd.TimeGrouper(wsize)).apply(corrR).reset_index(1, drop=True)
    #feats = feats.join(pairs_cor) 
    
    y = data['label'].resample(wsize, how=lambda ts: mode(ts)[0] if ts.shape[0] > 0 else np.nan)   

    
    # drop any nan values
    mask = np.any(np.isnan(feats), axis=1)
    feats, y = feats[~mask], y[~mask]
    mask = np.isnan(y)
    feats, y = feats[~mask], y[~mask]
    print('::::END:::: Get Simple Features::::')
    return (feats, y)


def train_model(X, y, est, grid):
    print('::::Train Model::::')
    gs = GridSearchCV(estimator=est, param_grid=grid, scoring='accuracy', cv=5, n_jobs=-1)
    gs = gs.fit(X, y)
    
    return (gs.best_estimator_, gs.best_params_)

def eval_model(mod, X_test, y_test, mod_name, plt_roc=True):
    print('::::Eval Model::::')
    y_prob = mod.predict_proba(X_test)
    y_test_bin = label_binarize(y_test, classes=[1,2,3,4,5,6,7])
    
    y_test_bin_ravel = y_test_bin.ravel()
    y_prob_ravel = y_prob.ravel()
    
    y_test_bin_ravel = y_test_bin_ravel[:len(y_prob_ravel)]
    
    fpr, tpr, _ = roc_curve(y_test_bin_ravel, y_prob_ravel)
    roc_auc = auc(fpr, tpr)
    
    if plt_roc:
        plt.plot(fpr, tpr, lw=2,
                 label='average ROC curve (auc=%0.2f), model: %s' % (roc_auc,mod_name))
        plt.legend(loc="lower right")

    y_pred = mod.predict(X_test)
    score = accuracy_score(y_true=y_test, y_pred=y_pred)
    print('Accuracy score on the test set: %.3f' %score)
    
    confusion_ma = confusion_matrix(y_true=y_test, y_pred=y_pred)
    confusion_ma = pd.DataFrame(confusion_ma, index=list(range(1,8)), columns=list(range(1,8)))
    print('Confusion Matrix...')
    print(confusion_ma)
    
    return (roc_auc, score)


data = read_data('1')
param_range = [0.0001, 0.001, 0.01, 0.1]

feats, y = get_simple_features(data, wsize='10s')
# split data into train and test sets

X_train, X_test, y_train, y_test = train_test_split(feats, y, test_size=.25, random_state=0, stratify=y)


print('Support Vector Machine')
svm_model, params = train_model(X_train, y_train, 
                    est=SVC(probability=True),
                    grid={'C': param_range, 'gamma': param_range, 'kernel': ['linear']})
eval_model(svm_model, X_test, y_test, 'SVC')
              

print('K-Nearest Neighbor')
knn_model, params = train_model(X_train, y_train, 
                    est=KNeighborsClassifier(),
                    grid={'n_neighbors':[5, 8, 10], 'weights':['uniform', 'distance']})
eval_model(knn_model, X_test, y_test,'KNN')


def get_advanced_features(data, wsize_sec, overlap=.5):
    print('::::START:::: Get Advance Features ::::')
    def rms(ts): return np.sqrt(np.mean(ts**2))

    wsize = int(10*wsize_sec)
    feats = data[['xL','yL','zL']].rolling(wsize,int(wsize/2)).mean().add_suffix('_mean_l')
    #feats = feats.join(feat)
    #feats = data[['xR','yR','zR']].rolling(wsize,int(wsize/2)).mean().add_suffix('_mean_r')
    #feats = feats.join(feat)
    #feat = data[['xR','yR','zR']].rolling(wsize, int(wsize/2)).std().add_suffix('_std_r')
    #feats = feats.join(feat)
    feat = data[['xL','yL','zL']].rolling(wsize, int(wsize/2)).std().add_suffix('_std_l')
    feats = feats.join(feat)
    
    feat = data[['xL','yL','zL']].rolling(wsize, int(wsize/2)).var().add_suffix('_var_l')
    feats = feats.join(feat)
    #feat = data[['xR','yR','zR']].rolling(wsize, int(wsize/2)).var().add_suffix('_var_r')
    #feats = feats.join(feat)
    
    feat = data[['xL','yL','zL']].rolling(wsize, int(wsize/2)).apply(rms).add_suffix('_rms_l')
    feats = feats.join(feat)
    #feat = data[['xR','yR','zR']].rolling(wsize, int(wsize/2)).apply(rms).add_suffix('_rms_r')
    #feats = feats.join(feat)
        
    mean_mag = (data**2).sum(axis=1).rolling(wsize, int(wsize/2)).apply(lambda ts: np.sqrt(ts).mean())
    mean_mag.name = 'mean_mag'
    feats = feats.join(mean_mag) 
    
    pairs_cor_ = data[['xL','yL','zL']].rolling(window=int(wsize/2)).corr(other=data[['xL','yL','zL']])
    feats = feats.join(pairs_cor_)
    #pairs_cor_ = data[['xR','yR','zR']].rolling(window=int(wsize/2)).corr(other=data[['xR','yR','zR']])
    #feats = feats.join(pairs_cor_)
    
    y = data[['label']].rolling(wsize, int(wsize/2)).apply(lambda ts: mode(ts)[0])  
    feats = feats.iloc[int(wsize*overlap)::int(wsize*overlap)] 
    
    y = y.iloc[int(wsize*overlap)::int(wsize*overlap)]
    print('::::END:::: Get features Advance::::')
    
    return feats, y

def get_advanced_features_predict(data, wsize_sec, overlap=.5):
    print('::::START:::: Get Advance Features ::::')
    def rms(ts): return np.sqrt(np.mean(ts**2))

    wsize = int(10*wsize_sec)
    feats = data[['xL','yL','zL']].rolling(wsize,int(wsize/2)).mean().add_suffix('_mean_l')
    
    #feats = feats.join(feat)
    #feats = data[['xR','yR','zR']].rolling(wsize,int(wsize/2)).mean().add_suffix('_mean_r')
    #feats = feats.join(feat)
    #feat = data[['xR','yR','zR']].rolling(wsize, int(wsize/2)).std().add_suffix('_std_r')
    #feats = feats.join(feat)
    feat = data[['xL','yL','zL']].rolling(wsize, int(wsize/2)).std().add_suffix('_std_l')
    feats = feats.join(feat)
    
    feat = data[['xL','yL','zL']].rolling(wsize, int(wsize/2)).var().add_suffix('_var_l')
    feats = feats.join(feat)
    #feat = data[['xR','yR','zR']].rolling(wsize, int(wsize/2)).var().add_suffix('_var_r')
    #feats = feats.join(feat)
    
    feat = data[['xL','yL','zL']].rolling(wsize, int(wsize/2)).apply(rms).add_suffix('_rms_l')
    feats = feats.join(feat)
    #feat = data[['xR','yR','zR']].rolling(wsize, int(wsize/2)).apply(rms).add_suffix('_rms_r')
    #feats = feats.join(feat)
        
    mean_mag = (data**2).sum(axis=1).rolling(wsize, int(wsize/2)).apply(lambda ts: np.sqrt(ts).mean())
    mean_mag.name = 'mean_mag'
    feats = feats.join(mean_mag) 
    
    pairs_cor_ = data[['xL','yL','zL']].rolling(window=int(wsize/2)).corr(other=data[['xL','yL','zL']])
    feats = feats.join(pairs_cor_)
    #pairs_cor_ = data[['xR','yR','zR']].rolling(window=int(wsize/2)).corr(other=data[['xR','yR','zR']])
    #feats = feats.join(pairs_cor_)
    
    
    feats = feats.iloc[int(wsize*overlap)::int(wsize*overlap)] 

    print('::::END:::: Get features Advance::::')
    
    
    return feats

win_sizes = ['2']#,'3', '5', '7', '10', '13', '15', '20']
best_model = RandomForestClassifier(criterion='entropy', n_jobs=-1, n_estimators=50)

for wsize in win_sizes:
    print('Window Size: %s sec' % wsize)
    print('Min periodos:', int(wsize)/2)
    try:
        # disjoint window
        print('Disjoint window:')
        feats, y = get_simple_features(data, wsize=wsize + 's')
               
        print(feats)
        X_train, X_test, y_train, y_test = train_test_split(feats, y, test_size=.25,
                                                            random_state=0, stratify=y)
        
        
        best_model.fit(X_train, y_train)
        roc_auc, acc = eval_model(best_model, X_test, y_test,'%ss - disjoint' %wsize, plt_roc=False)
        print('AUC score: %.3f' % roc_auc)


        # overlapping window
        print('Overlapping window:')
        feats, y = get_advanced_features(data, int(wsize))
        X_train, X_test, y_train, y_test = train_test_split(feats, y, test_size=.25,
                                                            random_state=0, stratify=y)
        
        best_model.fit(X_train, y_train)
        roc_auc, acc = eval_model(best_model, X_test, y_test,'%ss - overlapping' %wsize, plt_roc=False)
        print('AUC score: %.3f' % roc_auc)
    except Exception as e:
        print(e)

#data2 = read_data('all') 
#feats, y = get_advanced_features(data2, 2)
#X_train, X_test, y_train, y_test = train_test_split(feats, y, test_size=.25,
#                                                    random_state=0, stratify=y)
#X_train.fillna(X_train.mean())
#best_model.fit(X_train, y_train)
#eval_model(best_model, X_test, y_test,'2sec - overlapping')

#feats = get_advanced_features_predict(predict_data,2)
#print('::::FEATS::::::')
#feats.fillna(feats.mean())
#print(len(feats))

#print(':::::Predict SVM:::::::')
#re = svm_model.predict(feats)
#print(len(re))

#re = knn_model.predict(feats)
#print(':::::Predict KNN:::::::')
#print(len(re))



def predict_post(data):
    
    data = json.loads(data)
    
    print(data)

    freq = '19230U'
    data_frame = []
    for d in data:
        row = [d['index'],d['x'],d['y'],d['z']]
        data_frame += [row]
    pd_data_frame = pd.DataFrame(data_frame,columns=['index','xL','yL','zL'])
    pd_data_frame.index = pd.date_range(start='00:00:00', periods=pd_data_frame.shape[0], freq=freq)

    print(pd_data_frame.empty)
    if(not pd_data_frame.empty):
        feats = get_advanced_features_predict(pd_data_frame,2)

        print(feats)
        return svm_model.predict(feats)
    else:
        return 'Empty data frame'


#the callback function
def on_connect(client, userdata, flags, rc):
     print("Connected With Result Code {}".format(rc))
     client.subscribe("TestingTopic")

def on_disconnect(client, userdata, rc):
	print("Disconnected From Broker")

def on_message(client, userdata, message):
    print(predict_post(message.payload.decode()))

broker_address = "iot.eclipse.org"
#broker_address = "test.mosquitto.org"
broker_portno = 1883
client = mqtt.Client()

#Assigning the object attribute to the Callback Function
client.on_connect = on_connect
client.on_disconnect = on_disconnect
client.on_message = on_message

client.connect(broker_address, broker_portno)

client.loop_forever()